from PIL import Image

class TextOnlyLLM:
    # Reuses the same Qwen2.5-VL checkpoint already loaded elsewhere in this
    # project (no separate API key / service needed) purely as a text chat
    # model, for the mapper's schema-constrained JSON fallback stage.
    def __init__(self,model_name_or_path,device_map='auto'):
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.torch=torch; self.processor=AutoProcessor.from_pretrained(model_name_or_path)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path,torch_dtype='auto',device_map=device_map); self.model.eval()
    def generate(self,prompt,max_new_tokens=32,temperature=0.0):
        messages=[{'role':'user','content':[{'type':'text','text':prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],padding=True,return_tensors='pt'); device=next(self.model.parameters()).device
        inputs={k:(v.to(device) if hasattr(v,'to') else v) for k,v in inputs.items()}; kw={'max_new_tokens':max_new_tokens,'do_sample':temperature>0}
        if temperature>0:kw['temperature']=temperature
        with self.torch.no_grad():out=self.model.generate(**inputs,**kw)
        gen=out[:,inputs['input_ids'].shape[1]:]; return self.processor.batch_decode(gen,skip_special_tokens=True)[0].strip()

class QwenPairCaptioner:
    def __init__(self,model_name_or_path,device_map='auto'):
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.torch=torch; self.processor=AutoProcessor.from_pretrained(model_name_or_path)
        self.model=Qwen2_5_VLForConditionalGeneration.from_pretrained(model_name_or_path,torch_dtype='auto',device_map=device_map); self.model.eval()
    def generate(self,t1_path,t2_path,prompt,max_new_tokens=96,temperature=0.0):
        img1=Image.open(t1_path).convert('RGB'); img2=Image.open(t2_path).convert('RGB')
        messages=[{'role':'user','content':[{'type':'image','image':img1},{'type':'image','image':img2},{'type':'text','text':prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        inputs=self.processor(text=[text],images=[img1,img2],padding=True,return_tensors='pt'); device=next(self.model.parameters()).device
        inputs={k:(v.to(device) if hasattr(v,'to') else v) for k,v in inputs.items()}; kw={'max_new_tokens':max_new_tokens,'do_sample':temperature>0}
        if temperature>0:kw['temperature']=temperature
        with self.torch.no_grad():out=self.model.generate(**inputs,**kw)
        gen=out[:,inputs['input_ids'].shape[1]:]; return self.processor.batch_decode(gen,skip_special_tokens=True)[0].strip()
