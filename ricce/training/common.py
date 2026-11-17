from PIL import Image
import torch

def load_qwen_with_lora(model_name,r=16,alpha=32,dropout=0.05):
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    from peft import LoraConfig,get_peft_model
    p=AutoProcessor.from_pretrained(model_name); m=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,torch_dtype='auto',device_map='auto')
    cfg=LoraConfig(r=r,lora_alpha=alpha,lora_dropout=dropout,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    return get_peft_model(m,cfg),p

def load_merged(model_name,adapter_path=None):
    """Base model with a LoRA adapter merged in (or the bare base model if
    adapter_path is None), returned as a plain (non-PEFT) module. Used to
    build the frozen preference-optimization reference model from a
    completed SFT checkpoint, per the internal design review's requirement
    that DPO/TCPO use the SFT policy -- not the un-tuned base model -- as
    both the training start point and the reference distribution."""
    from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
    p=AutoProcessor.from_pretrained(model_name); m=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name,torch_dtype='auto',device_map='auto')
    if adapter_path:
        from peft import PeftModel
        m=PeftModel.from_pretrained(m,adapter_path); m=m.merge_and_unload()
    return m,p

def load_qwen_with_lora_from(model_name,adapter_path,r=16,alpha=32,dropout=0.05):
    """Trainable policy: base model with `adapter_path` merged in, then a
    *fresh* LoRA adapter applied on top for DPO/TCPO fine-tuning -- i.e.
    preference optimization continues from the SFT policy rather than
    restarting from the zero-shot base model."""
    from peft import LoraConfig,get_peft_model
    m,p=load_merged(model_name,adapter_path)
    cfg=LoraConfig(r=r,lora_alpha=alpha,lora_dropout=dropout,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'])
    return get_peft_model(m,cfg),p

def build_pair_inputs(processor,t1,t2,prompt,answer=None,max_length=1024):
    i1=Image.open(t1).convert('RGB'); i2=Image.open(t2).convert('RGB'); msgs=[{'role':'user','content':[{'type':'image','image':i1},{'type':'image','image':i2},{'type':'text','text':prompt}]}]
    pt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True); full=pt if answer is None else pt+answer
    enc=processor(text=[full],images=[i1,i2],padding=True,truncation=True,max_length=max_length,return_tensors='pt'); labels=enc['input_ids'].clone()
    if answer is not None:
        penc=processor(text=[pt],images=[i1,i2],padding=True,truncation=True,max_length=max_length,return_tensors='pt'); plen=min(penc['input_ids'].shape[1],labels.shape[1]); labels[:,:plen]=-100
    enc['labels']=labels; return enc

def build_pair_inputs_batch(processor,items,prompt,max_length=1024):
    """True multi-sample batching for the DPO/TCPO loop ('proper
    batching' fix -- the original per-sample micro-batch=1 loop only ever
    relied on gradient *accumulation* to reach an effective batch size,
    never an actual batched forward pass). `items` is a list of
    (t1_path, t2_path, answer_or_None) tuples; returns one collated batch
    dict with `labels` masked per-sample to only the answer span, exactly
    like the single-sample `build_pair_inputs`, just for B samples at once.
    Right-padding (the processor's tokenizer default here) makes each
    sample's true prompt length -- computed from an unpadded per-sample
    encoding, not sliced out of the batch-padded prompt-only encoding --
    the correct mask boundary regardless of what any other sample in the
    batch pads to."""
    imgs1=[Image.open(t1).convert('RGB') for t1,t2,a in items]
    imgs2=[Image.open(t2).convert('RGB') for t1,t2,a in items]
    texts_full=[];all_images=[];prompt_lens=[]
    for (t1,t2,a),i1,i2 in zip(items,imgs1,imgs2):
        msgs=[{'role':'user','content':[{'type':'image','image':i1},{'type':'image','image':i2},{'type':'text','text':prompt}]}]
        pt=processor.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True)
        texts_full.append(pt if a is None else pt+a)
        all_images+=[i1,i2]
        penc=processor(text=[pt],images=[i1,i2],padding=False,truncation=True,max_length=max_length,return_tensors='pt')
        prompt_lens.append(penc['input_ids'].shape[1])
    enc=processor(text=texts_full,images=all_images,padding=True,truncation=True,max_length=max_length,return_tensors='pt')
    labels=enc['input_ids'].clone()
    for i,plen in enumerate(prompt_lens):
        plen=min(plen,labels.shape[1]);labels[i,:plen]=-100
    if 'attention_mask' in enc:
        labels[enc['attention_mask']==0]=-100
    enc['labels']=labels
    return enc


def sequence_logprob(model,batch):
    labels=batch['labels']; clean=labels.clone(); clean[clean==-100]=0; out=model(**{k:v for k,v in batch.items() if k!='labels'}); logits=out.logits[:,:-1,:]; target=clean[:,1:]; mask=(labels[:,1:]!=-100); logp=torch.log_softmax(logits,dim=-1); tok=logp.gather(-1,target.unsqueeze(-1)).squeeze(-1); return (tok*mask).sum(-1)/mask.sum(-1).clamp_min(1)
