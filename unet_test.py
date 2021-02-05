import h5py
import numpy as np
import PIL
import subprocess
import os
from PIL import Image
import sys
from tifffile import imread, TiffFile
from bfio import BioReader, BioWriter, LOG4J, JARS
import javabridge, math
from pathlib import Path
from multiprocessing import cpu_count


def rescale(size,img,mode='uint8'):
    
    if mode == 'float32':
        #for floating point images:
        img = np.float32(img)
        img_PIL = PIL.Image.fromarray(img,mode='F')
    elif mode == 'uint8':
        #otherwise:
        img_PIL = PIL.Image.fromarray(img)
    else:
        raise(Exception('Invalid rescaling mode. Use uint8 or float32'))
          
    return np.array(img_PIL.resize(size,PIL.Image.BILINEAR))


def normalize(img):

    ###normalize image
    img_min = np.min(img)
    img_max = np.max(img)
    img_centered = img - img_min
    img_range = img_max - img_min
    return np.true_divide(img_centered, img_range)


def unet_segmentation(ind,input_img,img_pixelsize_x,img_pixelsize_y,
                          modelfile_path,weightfile_path,iofile_path,
                          tiling_x=4,tiling_y=4,gpu_flag='0',
                          cleanup=True):

    #fix parameters
    n_inputchannels=1
    n_iterations=0
    
    
    ## prepare image rescaling
    np.set_printoptions(threshold=sys.maxsize)

    #get model resolution (element size) from modelfile
    modelfile_h5 = h5py.File(modelfile_path,'r')
    modelresolution_y = modelfile_h5['unet_param/element_size_um'][0]
    modelresolution_x = modelfile_h5['unet_param/element_size_um'][1]
    modelfile_h5.close()       
    #get input image absolute size
    abs_size_x = input_img.shape[1] * img_pixelsize_x
    abs_size_y = input_img.shape[0] * img_pixelsize_y
    #get rescaled image size in pixel
    rescaled_size_px_x = int(np.round(abs_size_x / modelresolution_x))
    rescaled_size_px_y = int(np.round(abs_size_y / modelresolution_y))
    rescale_size = (rescaled_size_px_x,rescaled_size_px_y)
    ### preprocess image and store in IO file

    #normalize image, then rescale
    normalized_img = normalize(input_img)
    rescaled_img = np.float32(rescale(rescale_size,normalized_img,mode='float32'))
    #prepending singleton dimensions to get the desired blob structure
    h5ready_img = np.expand_dims(rescaled_img, axis=(0,1))
    iofile_h5 = h5py.File(iofile_path,mode='x')
    iofile_h5.create_dataset('data',data=h5ready_img)
    iofile_h5.close()

    # ### run caffe_unet commands

    # #assemble sanity check command
    command_sanitycheck = []
    command_sanitycheck.append("caffe_unet")
    command_sanitycheck.append("check_model_and_weights_h5")
    command_sanitycheck.append("-model")
    command_sanitycheck.append(modelfile_path)
    command_sanitycheck.append("-weights")
    command_sanitycheck.append(weightfile_path)
    command_sanitycheck.append("-n_channels")
    command_sanitycheck.append(str(n_inputchannels))
    if gpu_flag:
        command_sanitycheck.append("-gpu")
        command_sanitycheck.append(gpu_flag)
     #runs command and puts console output to stdout
    sanitycheck_proc = subprocess.run(command_sanitycheck,stdout=subprocess.PIPE)
    # #aborts if process failed
    sanitycheck_proc.check_returncode()

    #assemble prediction command
    command_predict = []
    command_predict.append("caffe_unet")
    command_predict.append("tiled_predict")
    command_predict.append("-infileH5")
    command_predict.append(iofile_path)
    command_predict.append("-outfileH5")
    command_predict.append(iofile_path)
    command_predict.append("-model")
    command_predict.append(modelfile_path)
    command_predict.append("-weights")
    command_predict.append(weightfile_path)
    command_predict.append("-iterations")
    command_predict.append(str(n_iterations))
    command_predict.append("-n_tiles")
    command_predict.append(str(tiling_x)+'x'+str(tiling_y))
    command_predict.append("-gpu")
    command_predict.append(gpu_flag)
    if gpu_flag:
        command_predict.append("-gpu")
        command_predict.append(gpu_flag)
    #run command 
    output = subprocess.check_output(command_predict, stderr=subprocess.STDOUT).decode()
    print(output)

    # load results from io file and return
    output_h5 = h5py.File(iofile_path)
    score = output_h5['score'][:]
    output_h5.close()
    # #get segmentation mask by taking channel argmax
    segmentation_mask = np.squeeze(np.argmax(score, axis=1))
    return segmentation_mask


def run_segmentation(ome_path, tif, ind, pixelsize, output_directory):

    out_path = Path(output_directory)
    br = BioReader(ome_path,backend='java')
    with BioWriter(out_path.joinpath(f"output{ind}.ome.tif"),metadata = br.metadata,
                            backend='java') as bw:
        for page in tif.pages:
                image = page.asarray()
                for x_ in range(0,image.shape[1],1024):
                    x_max = min([image.shape[1],x_+1024])
                    for y_ in range(0,image.shape[0], 1024):
                        y_max = min([image.shape[0],y_+1024])
                        input_img = image[x_:x_+1024, y_:y_+1024]
                        img_pixelsize_x = pixelsize                 
                        img_pixelsize_y = pixelsize
                        modelfile_path = "2d_cell_net_v0-cytoplasm.modeldef.h5"
                        weightfile_path = "snapshot_cytoplasm_iter_1000.caffemodel.h5"
                        iofile_path = "output.h5"
                        img = unet_segmentation(ind,input_img,img_pixelsize_x, \
                                       img_pixelsize_y,modelfile_path,weightfile_path,iofile_path)
                        bw[y_:y_max, x_:x_max,...] = img
                        os.remove("output.h5")
    br.close()
    os.remove(f"out{ind}.ome.tif")


def read_file(input_directory, pixelsize, output_directory):

    rootdir = Path(input_directory)
    """ Convert the tif to tiled tiff """
    javabridge.start_vm(args=["-Dlog4j.configuration=file:{}".format(LOG4J)],
                        class_path=JARS,
                        run_headless=True)
    i = 0
    try:
        for PATH in rootdir.glob('**/*'):
            if str(PATH).find("ome.tif")== -1:
                tile_grid_size = 1
                tile_size = tile_grid_size * 1024

                # Set up the BioReader
                with BioReader(PATH,backend='java',max_workers=cpu_count()) as br:
 
                    # Loop through timepoints
                    for t in range(br.T):

                        # Loop through channels
                        for c in range(br.C):

                            with BioWriter(f'out{i}.ome.tif',
                                    backend='java',
                                    metadata=br.metadata,
                                    max_workers = cpu_count()) as bw:

                                 # Loop through z-slices
                                for z in range(br.Z):

                                    # Loop across the length of the image
                                    for y in range(0,br.Y,tile_size):
                                        y_max = min([br.Y,y+tile_size])

                                        # Loop across the depth of the image
                                        for x in range(0,br.X,tile_size):
                                            x_max = min([br.X,x+tile_size])
                                            bw[y:y_max,x:x_max,z:z+1,0,0] = br[y:y_max,x:x_max,z:z+1,c,t]
 
                            with TiffFile(f'out{i}.ome.tif') as tif:
                                    ome_path = f'out{i}.ome.tif'
                                    run_segmentation(ome_path,tif, i, pixelsize, output_directory)
                            
            elif str(PATH).find("ome.tif") != -1:
                with TiffFile(PATH) as tif:
                    run_segmentation(PATH,tif, i, pixelsize, output_directory)
            i+=1

    finally:
        # Close the javabridge. Since this is in the finally block, it is always run
        javabridge.kill_vm()