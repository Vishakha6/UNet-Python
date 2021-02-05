# from bfio.bfio import BioReader, BioWriter
# import bioformats
# import javabridge as jutil
import argparse, logging, subprocess, time, multiprocessing, sys
# import numpy as np
from pathlib import Path
import unet_test

if __name__=="__main__":
    # Initialize the logger
    logging.basicConfig(format='%(asctime)s - %(name)-8s - %(levelname)-8s - %(message)s',
                        datefmt='%d-%b-%y %H:%M:%S')
    logger = logging.getLogger("main")
    logger.setLevel(logging.INFO)

    ''' Argument parsing '''
    logger.info("Parsing arguments...")
    parser = argparse.ArgumentParser(prog='main', description='A plugin to test the UNet model by U-Freiburg.')
    
    # Input arguments
    parser.add_argument('--input_directory', dest='input_directory', type=str,
                        help='Input image collection to be processed by this plugin', required=True)
    parser.add_argument('--pixelsize', dest='pixelsize', type=int,
                        help='Input image pixel size', required=True)
    # Output arguments
    parser.add_argument('--output_directory', dest='output_directory', type=str,
                        help='Output collection', required=True)
    
    # Parse the arguments
    args = parser.parse_args()
    input_directory = args.input_directory
    if (Path.is_dir(Path(args.input_directory).joinpath('images'))):
        # switch to images folder if present
        fpath = str(Path(args.input_directory).joinpath('images').absolute())
    logger.info('input_directory = {}'.format(input_directory))
    pixelsize = args.pixelsize
    logger.info('pixelsize = {}'.format(pixelsize))
    output_directory = args.output_directory
    logger.info('output_directory = {}'.format(output_directory))
    unet_test.read_file(input_directory, pixelsize, output_directory)
    # # Surround with try/finally for proper error catching
    # try:
    #     # Start the javabridge with proper java logging
    #     logger.info('Initializing the javabridge...')
    #     log_config = Path(__file__).parent.joinpath("log4j.properties")
    #     jutil.start_vm(args=["-Dlog4j.configuration=file:{}".format(str(log_config.absolute()))],class_path=bioformats.JARS)
    #     # Get all file names in input_directory image collection
    #     input_directory_files = [f.name for f in Path(input_directory).iterdir() if f.is_file() and "".join(f.suffixes)=='.ome.tif']
        
        
    #     # Loop through files in input_directory image collection and process
    #     for i,f in enumerate(input_directory_files):
    #         # Load an image
    #         br = BioReader(Path(input_directory).joinpath(f))
    #         image = np.squeeze(br.read_image())

    #         # initialize the output
    #         out_image = np.zeros(image.shape,dtype=br._pix['type'])

    #         """ Do some math and science - you should replace this """
    #         logger.info('Processing image ({}/{}): {}'.format(i,len(input_directory_files),f))
    #         out_image = awesome_math_and_science_function(image)

    #         # Write the output
    #         bw = BioWriter(Path(output_directory).joinpath(f),metadata=br.read_metadata())
    #         bw.write_image(np.reshape(out_image,(br.num_y(),br.num_x(),br.num_z(),1,1)))
        
    # finally:
    #     # Close the javabridge regardless of successful completion
    #     logger.info('Closing the javabridge')
    #     jutil.kill_vm()
        
    #     # Exit the program
    #     sys.exit()