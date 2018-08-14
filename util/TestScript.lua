require('torch')
require('nngraph')
require('image')
--require('fbtorch')
require('gfx.js')
require('cunn')
require('cudnn')

paths.dofile('upsample.lua')
paths.dofile('expand.lua')
--dofile('ucf101.lua')

torch.manualSeed(1)
torch.setnumthreads(4)
iscuda = false
assert(loadfile("image_error_measures.lua"))(iscuda)

opt_default = {
   full = false, -- display previous frames and target, otherwise the prediction
   with_pyr = true,
   with_delta = true,
   with_cuda = true,
   network_dir = 'AdvGDL',
   delay_gif = 25,
   totalNbiters=1,
   nChannels= 3,
   margin = 5, --for display
   nOutputFrames = 1,
   nOutputFramesRec = 1,--3 for MultiStep
   interv = 1,
   flow_im_used=true
}

op = op or {}
for k, v in pairs(opt_default) do
   if op[k] == nil then
      op[k] = v
   end
end

local inputH, inputW = 240, 320
local netsize = 64
opt = {batchsize = 1}

-- loading trained network

local flow_pth = '/home/abhishek/Workspace/VideoPredictionICLR2016/MathieuICLR16TestCode (1)/MathieuICLR16TestCode/UCF101frm10p/'
local predloaded
if op.network_dir=='Adv'  then
  predloaded = torch.load('/home/abhishek/Workspace/VideoPredictionICLR2016/MathieuICLR16TestCode (1)/MathieuICLR16TestCode/trained_models/new_adv_big_64_smalladv.t7')
elseif op.network_dir=='AdvGDL'  then
  predloaded = torch.load('/home/abhishek/Workspace/VideoPredictionICLR2016/MathieuICLR16TestCode (1)/MathieuICLR16TestCode/trained_models/new_adv_big_gdl_64.t7')
end
local opt = predloaded.opt
local model = predloaded.model
opt.nOutputFrames = 1
opt.batchsize = 1

------------------------------------------------------------------------------
-- init multiscale model with dsnet
   local dsnet = nn.ConcatTable()
   dsnet:add(nn.SpatialAveragePooling(8,8,8,8))
   dsnet:add(nn.SpatialAveragePooling(4,4,4,4))
   dsnet:add(nn.SpatialAveragePooling(2,2,2,2))
   dsnet:add(nn.SpatialAveragePooling(1,1,1,1))
   dsnet:cuda()
   local dsnetInput = dsnet
   local dsnetTarget = dsnet:clone()

--------------------------------------------------------------------------------
-- network size adaptation for models fine-tuned on larger patchs
for i = 1, #model.modules do
  if torch.type(model.modules[i]) == 'nn.ExpandDim' then
    local xH = math.floor(math.sqrt(model.modules[i].k) /netsize * inputH + 0.5)
    local xW = math.floor(math.sqrt(model.modules[i].k) /netsize * inputW + 0.5)
        model.modules[i].k = xH*xW
    end
    if torch.type(model.modules[i]) == 'nn.View' then
      if model.modules[i].numInputDims == 2 then
        local s1 = model.modules[i].size[1]
        local s2 = math.floor(model.modules[i].size[2] /netsize * inputH + 0.5)
        local s3 = math.floor(model.modules[i].size[3] /netsize * inputW + 0.5)
        model.modules[i].size = torch.LongStorage{s1, s2, s3}
        model.modules[i].numElements = s1*s2*s3
        --print(model.modules.size)
      end
    end
end

local delta = {torch.CudaTensor(opt.batchsize, 2):zero(),
               torch.CudaTensor(opt.batchsize, 4):zero(),
               torch.CudaTensor(opt.batchsize, 6):zero(),
               torch.CudaTensor(opt.batchsize, 8):zero()}

------------------------------------------------------------------------------

function display_frames(my_array,nbframes)

   local inter = torch.Tensor(op.nChannels,my_array:size(2),op.margin):fill(1)
   local todisp = torch.Tensor(op.nChannels,my_array:size(2),op.margin):fill(1)
   local todisp2 = torch.Tensor(nbframes,op.nChannels,my_array:size(2),
      my_array:size(3))
   for i = 1, nbframes do
      for j = 1, op.nChannels do
         todisp2[i][j]= my_array[(i-1)*3+j]
      end
      todisp = torch.cat(todisp, todisp2[i], 3)
      todisp = torch.cat(todisp, inter, 3)
    end
   gfx.image(todisp)
end

function save_frames(prediction, nbframes, filename)
   for i = 1, opt.nInputFrames do
      prediction[i]:add(1):div(2)

      image.save(filename..'/pred_'..i..'.png',prediction[i])
    end
    local new_img = torch.Tensor(op.nChannels,inputH, inputW):fill(0)
    new_img[1]:fill(1)
    for i = opt.nInputFrames+1, opt.nInputFrames+op.nOutputFramesRec do
      prediction[i]:add(1):div(2)
      new_img[{{},{3,inputH-2},{3,inputW-2}}]=
        prediction[i][{{},{3,inputH-2},{3,inputW-2}}]
      image.save(filename..'/pred_'..i..'.png',new_img)
    end
end

------------------------------------------------------------------------------
-- Main job

local sum_PSNR=torch.Tensor(op.nOutputFramesRec):fill(0)
local sum_err_sharp2=torch.Tensor(op.nOutputFramesRec):fill(0)
local sum_SSIM=torch.Tensor(op.nOutputFramesRec):fill(0)
local sum_PSNRwarp=torch.Tensor(op.nOutputFramesRec):fill(0)
local sum_err_sharp2warp=torch.Tensor(op.nOutputFramesRec):fill(0)
local sum_SSIMwarp=torch.Tensor(op.nOutputFramesRec):fill(0)
local nbimagestosave = op.nOutputFramesRec+opt.nInputFrames
local array_to_save= torch.Tensor(nbimagestosave,op.nChannels,inputH,inputW)
local target_to_save =
  torch.Tensor(op.nOutputFramesRec,op.nChannels,inputH,inputW)

local input, output, target
local batch=1
local nbvideos = 3783
local nbframes, nbpartvid
local nbvid = torch.Tensor(op.nOutputFramesRec):fill(0)

local index =
  torch.range(1,(opt.nInputFrames+op.nOutputFramesRec)*op.interv, op.interv)

local psnr_tot = 0
local ssim_tot = 0
for videoidx = 1,nbvideos,10 do


    local filename_out = op.network_dir..'/'..videoidx
    for ii = 1,op.nOutputFramesRec do

        output = image.load('/home/abhishek/Workspace/UCF_DyanOF/Train/UCF4/l11p40/Lpoint1/Test/c2f_3step/Results_Last_3Step_C2F/'..videoidx..'/PDFrame-000'..(ii+4)..'.png')
        
        output = output[{{1,3}}]
      
        output = image.scale(output,256,256)
        output:mul(2):add(-1)

        target = image.load('/home/abhishek/Downloads/MathieuICLR16TestCode/MathieuICLR16TestCode/UCF101frm10p/'..videoidx..'/target_'..(ii)..'.png')
        target = target[{{1,3}}]
        target = image.scale(target,256,256)
        target:mul(2):add(-1)
      
        if op.flow_im_used then
            local flow_im_name
            local moutput = torch.Tensor(3,256,256):fill(-1)
            local mtarget = torch.Tensor(3,256,256):fill(-1)
            
            flow_im_name = '/home/abhishek/Downloads/motion_masks_ucf101_interp/'..videoidx..'/motion_mask.png'
            
            local flow_im = image.load(flow_im_name)
            
            for j=1, 256 do
              for k=1, 256 do
                  if flow_im[1][j][k] == 1 then
                    for i=1,3 do
                      moutput[i][j][k] = output[i][j][k]
                      mtarget[i][j][k] = target[i][j][k]
                    end
                end
              end
            end
            
            local psnr = PSNR(mtarget, moutput)
            

            if psnr < 50 then
                
                sum_PSNR[ii] = sum_PSNR[ii]+psnr
                sum_SSIM[ii] = sum_SSIM[ii]+SSIM(moutput, mtarget)
                sum_err_sharp2[ii] = sum_err_sharp2[ii] +
                  computel1difference(moutput, mtarget)
                nbvid[ii] = nbvid[ii]+1
            end
        else
          sum_PSNR[ii] = sum_PSNR[ii]+PSNR(output[{{1,3}}], target[{{1,3}}])
          sum_SSIM[ii] = sum_SSIM[ii]+SSIM(output[{{1,3}}], target[{{1,3}}])
          sum_err_sharp2[ii] = sum_err_sharp2[ii] +
            computel1difference(output[{{1,3}}], target[{{1,3}}])

          nbvid[ii] = nbvid[ii]+1

        
        end
    end
    

    print(filename_out)
    os.execute('mkdir -p "' .. filename_out .. '"; ')
    save_frames(array_to_save, nbimagestosave, filename_out)

    for i= 1,op.nOutputFramesRec do
     print('******** video '..videoidx..', '..i..' th frame pred *************')
     print(string.format("score sharp diff: %.2f",sum_err_sharp2[i]/nbvid[i]))
     print(string.format("PSNR: %.2f",sum_PSNR[i]/nbvid[i]))
     print(string.format("SSIM: %.2f",sum_SSIM[i]/nbvid[i]))

    end

end