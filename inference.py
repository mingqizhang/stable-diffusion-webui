from glob import glob
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from matplotlib import image
from pydantic import StrError
from webui import initialize
from modules.processing import StableDiffusionProcessingTxt2Img, \
StableDiffusionProcessingImg2Img, process_images, process_images_inner
import modules.shared as shared
from modules.shared import opts, cmd_opts, extensions
from modules import extra_networks, extra_networks_hypernet
import modules.sd_models
import modules.script_callbacks
import modules.scripts
from PIL import Image
from modules import sd_hijack

def SampleMethod(Method: str='Euler a'):
    samples = {
        'Euler a'           :0, # default
        'Euler'             :1,
        'LMS'               :2,
        'Heun'              :3,
        'DPM2'              :4,
        'DPM2 a'            :5,
        'DPM++ 2S a'        :6,
        'DPM++ 2M'          :7,
        'DPM fast'          :8,
        'DPM adaptive'      :9,
        'LMS Karras'        :10,
        'DPM2 Karras'       :11,
        'DPM2 a Karras'     :12,
        'DPM++ 2S a Karras' :13,
        'DPM++ 2M Karras'   :14,
        'DDIM'              :15,
        'PLMS'              :16,
    }
    if Method not in samples:
        raise StrError("Place choice the correct sample method!")
    else:
        return samples[Method]

def ResizeMode(Method: str='Just resize'):
    Modes = {
        'Just resize'       :0,
        'Crop and resize'   :1,
        'Resize and fill'   :2,
    }
    if Method not in Modes:
        raise StrError("Place choice the correct resize mode!")
    else:
        return Modes[Method]


class StableDiffusion():
    def __init__(self, modelPath: str, vaepath: str=None, precision: str="autocast", no_half: bool=False, no_half_vae: bool=True):
        super().__init__
        self.default_prompt = "masterpiece,best quality"
        self.default_negative_prompt = "nsfw,lowers,bad anatomy, bad hands, text,error,missing fingers,extra digit,\
                                        fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,\
                                        signature,watermark,username,blurry"
        shared.cmd_opts.ckpt = modelPath
        shared.cmd_opts.vae_path = vaepath if vaepath is not None else None
        shared.cmd_opts.no_half = no_half
        shared.cmd_opts.precision = precision
        shared.cmd_opts.no_half_vae = no_half_vae
    def init(self, ):
        extensions.list_extensions()
        modules.scripts.load_scripts()  # load lora extensions-builtin
        modules.sd_models.setup_model()
        modules.sd_models.load_model()
        sd_hijack.model_hijack.embedding_db.load_textual_inversion_embeddings()
        extra_networks.initialize()
        extra_networks.register_extra_network(extra_networks_hypernet.ExtraNetworkHypernet())
        modules.script_callbacks.before_ui_callback()  # add lora 
    def text2img(self, prompt: str="" , 
                        savePath: str='', 
                        negative_prompt: str="", 
                        seed: int=1, 
                        samplemethod: str='Euler a',
                        steps: int=20, 
                        width: int=512, 
                        height: int=768, 
                        cfg_scale: int=12, 
                        batch_size:int=1, 
                        n_iter: int=1, ):
        
        if savePath is None:
            raise StrError("save path is error!")
        p = StableDiffusionProcessingTxt2Img(
                sd_model=shared.sd_model,
                outpath_samples=savePath,
                outpath_grids=opts.outdir_grids or opts.outdir_txt2img_grids,
                prompt= prompt if prompt else self.default_prompt,
                styles=[None, None],       # [None, None]
                negative_prompt=negative_prompt if negative_prompt else self.default_negative_prompt,
                seed=seed,
                sampler_index=SampleMethod(samplemethod),   # 0
                batch_size=batch_size,  
                n_iter=n_iter,                              # batch count
                steps=steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
        )
        res = process_images_inner(p)
        p.close()

    def img2img(self, imagePath: str='', 
                        savePath: str='', 
                        prompt: str="" , 
                        negative_prompt: str="", 
                        resize_mode: str="Just resize",
                        seed: int=1, 
                        samplemethod: str='Euler a',
                        steps: int=20, 
                        width: int=512, 
                        height: int=768, 
                        cfg_scale: int=12, 
                        batch_size:int=1, 
                        n_iter: int=1, 
                        denoising_strength:float=0.5):
        if imagePath is None:
            raise StrError("image path is error!")
        if savePath is None:
            raise StrError("save path is error!")
        try:
            self.image = Image.open(imagePath)
        except IOError:
            raise ValueError("read image error, please check the url!")
        p = StableDiffusionProcessingImg2Img(
                resize_mode=ResizeMode(resize_mode),
                sd_model=shared.sd_model,
                outpath_samples=savePath,
                outpath_grids=opts.outdir_grids or opts.outdir_img2img_grids,
                prompt=prompt if prompt else self.default_prompt,
                negative_prompt=negative_prompt if negative_prompt else self.default_negative_prompt,
                styles=[None, None],
                seed=seed,
                sampler_index=SampleMethod(samplemethod),     
                batch_size=batch_size,
                n_iter=n_iter,
                steps=steps,
                cfg_scale=cfg_scale,
                width=width,
                height=height,
                init_images=[self.image],
                denoising_strength=denoising_strength,
            )
        res = process_images_inner(p)
        p.close()

if __name__ == "__main__":
    
    sd = StableDiffusion(modelPath='./models/Stable-diffusion/chillout/chilloutmix_NiCkpt.ckpt',
                        vaepath="./models/VAE/vae-ft-mse-840000-ema-pruned.safetensors")
    sd.init()
    sd.text2img(prompt="masterpiece,best quality <lora:ym_v5:0.66>",
                negative_prompt='nsfw,lowers,bad anatomy, bad hands, text,error,missing fingers,\
                                extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,\
                                signature,watermark,username,blurry',
                savePath='./testttt.jpg',
                samplemethod = 'Euler a',
                seed=10,
                steps=20,
                cfg_scale=7,
                width=512,
                height=768,
                )
   
    # sd.img2img(prompt="masterpiece, best quality, man, male,  illustration, delicate details, refined rendering, extremely detailed CG unity 8k wallpaper, super strong muscles, muscle, Muscle strengthening, ", 
    #         negative_prompt = 'nsfw,lowers,bad anatomy, bad hands, text,error,missing fingers,extra digit,fewer digits,cropped,worst quality,low quality,normal quality,jpeg artifacts,signature,watermark,username,blurry',
    #         imagePath = imgfile if i==0 else os.path.join(path1, str(i-1)+'.jpg'),
    #         savePath = os.path.join(path1, str(i)+'.jpg'), 
    #         resize_mode = 'Resize and fill',
    #         samplemethod = 'Euler a',
    #         steps = 28,
    #         cfg_scale=20,
    #         seed=-1, 
    #         width=512,
    #         height=768, 
    #         denoising_strength=den)
       