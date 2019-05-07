#define __CL_ENABLE_EXCEPTIONS

#include <CL/cl.hpp>
#include <fstream>
#include <iostream>
#include <vector>
#include "model.h"

#include "Utils/utils.h"

static const char* input_imagePath = "images/ship.txt";

std::vector<std::string> split(const std::string& s, char c) {
  std::vector<std::string> v;
  unsigned int i = 0;
  unsigned int j = s.find(c);
  while (j < s.size() ){
    v.push_back(s.substr(i, j - i));
    i = ++j;
    j = s.find(c, j);
    if (j >= s.size() ) {
      v.push_back(s.substr(i,s.size() ));
      break;
    }
  }
  return v;
}
int main()
{
	float* hInputImage;
    float *w[NUMBER_WEIGHTS];
    for(int i=0;i<NUMBER_WEIGHTS;i++){
      w[i] = new float [weights[i]];	
  }

  std::cout<<"Loading weights into DDR memory"<<std::endl;
  std::ifstream in("weights_radar.dat",std::ios_base::binary);
  std::cout<<"Initializing weight buffers for  each layers"<<std::endl;

   for (int i=0;i<NUMBER_WEIGHTS;i++)
    {
      // std::cout<<"Yes";
      in.read((char *)w[i],sizeof(float)*weights[i]);
    }



	/* ------------------------------------- OpenCL Setup Code ------------------------------------- */
	
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	std::vector<cl::Device> devices;
	platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);

	cl::Context context(devices);

	cl::CommandQueue queue = cl::CommandQueue(context, devices[0], CL_QUEUE_PROFILING_ENABLE);

	float *output_buffer = new float [OUTPUT_BUFFER_SIZE];
	float *input_buffer = new float [INPUT_BUFFER_SIZE];
    for (int i =0;i<OUTPUT_BUFFER_SIZE;i++)
    {
    	output_buffer[i] = 0;
    }
    
   std::fstream inputi;
    std::fstream input_label;
    inputi.open("test.dat");
    input_label.open("test_label.dat");

    std::string lincA;
    std::string lincB;
     int count = 0;
     int testcases = 120;
    for (int i = 0;i<testcases;i++)
    {
    getline(inputi, lincA);
    getline(input_label,lincB);
    std::vector<std::string> listfilemax = split(lincA,',');
    int test_label = atof(lincB.c_str());
    for (int l=0; l<128*384*1; l++)
        {
            input_buffer[l]= atof(listfilemax.at(l).c_str());
            // std::cout<<input_buffer[l];

        }

   // std::cout<<"Done"  
 int weight_count = 0;
 for (int j =0;j<NUMBER_LAYERS;j++)
 {
   if (layer[j][0]==0)
   
   {
      	
	/* ------------------------------------ Layer 1 Starts ------------------------------------ */

	int in_channels, out_channels, kernel_size;
	in_channels = layer[j][1];
	out_channels = layer[j][2];
	kernel_size = layer[j][7];
	imgRows = layer[j][5];
	imgCols = layer[j][6];		
	std::cout<<"Performing Convolution  "<<j+1<<" "<<std::endl;
	// std::cout<<in_channels<<" "<<out_channels<<" "<<kernel_size<<" "<<imgRows;
	
  
	try
	{
		  cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*imgRows*imgCols*sizeof(float));
    cl::Buffer filterBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_channels*out_channels*kernel_size*kernel_size*sizeof(float));
    cl::Buffer biasBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_channels*sizeof(float));
    cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_channels*imgRows*imgCols*sizeof(float));
    cl::Buffer in_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer out_channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer kernelSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer imgRowsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer imgColsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    // std::cout<<"uoioj";
		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_channels*imgRows*imgCols*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(filterBuffer, CL_TRUE, 0, in_channels*out_channels*kernel_size*kernel_size*sizeof(float), w[weight_count]);
		queue.enqueueWriteBuffer(biasBuffer, CL_TRUE, 0, out_channels*sizeof(float), w[weight_count+1]);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(in_channelsBuffer, CL_TRUE, 0, sizeof(int), &in_channels);
		queue.enqueueWriteBuffer(out_channelsBuffer, CL_TRUE, 0, sizeof(int), &out_channels);
		queue.enqueueWriteBuffer(kernelSizeBuffer, CL_TRUE, 0, sizeof(int), &kernel_size);
		queue.enqueueWriteBuffer(imgRowsBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(imgColsBuffer, CL_TRUE, 0, sizeof(int), &imgCols);
   
 // std::cout<<"uoioj";
		std::ifstream sourceFile("cl_kernels/conv.cl");
        std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
         cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "convolution");

     	kernel.setArg(0, out_channelsBuffer);
     	kernel.setArg(1, in_channelsBuffer);
     	kernel.setArg(2, kernelSizeBuffer);
     	kernel.setArg(3, inputBuffer);
     	kernel.setArg(4, filterBuffer);
     	kernel.setArg(5, biasBuffer);
     	kernel.setArg(6, outputBuffer);
     	kernel.setArg(7, imgRowsBuffer);
     	kernel.setArg(8, imgColsBuffer);

     	cl::NDRange global(imgCols, imgRows);
     	cl::NDRange local(1, 1);
      // cl::Event event;
     	queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL);
      // queue.finish();
     	// Read data back
     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_channels*imgRows*imgCols*sizeof(float), output_buffer);
     // std::cout<<output_buffer[0];
	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
  // exit(0);
    weight_count = weight_count+2;
    for (int p = 0;p<(out_channels*imgRows*imgCols);p++)
          { 
               
              input_buffer[p] = output_buffer[p]; 
      }
    }
   
    else if (layer[j][0]==1)
      {
  
	/* ------------------------------------ Layer 1 Ends ------------------------------------ */

	
	/* ------------------------------------ MaxPool 2D Starts ------------------------------------ */

	int channels, pool_size, outImgRows, outImgCols;
	channels = layer[j][1];
	imgRows = layer[j][3];
	imgCols = layer[j][4];
	pool_size = 2;

	outImgRows = layer[j][5];
	outImgCols = layer[j][6];
	for (int i =0;i<channels*outImgRows*outImgCols;i++)
 {
  output_buffer[i] = 0;
 }


	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, channels*imgRows*imgCols*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, channels*outImgRows*outImgCols*sizeof(float));
		cl::Buffer channelsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer poolSizeBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer inDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outDimBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, channels*imgRows*imgCols*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(channelsBuffer, CL_TRUE, 0, sizeof(int), &channels);
		queue.enqueueWriteBuffer(poolSizeBuffer, CL_TRUE, 0, sizeof(int), &pool_size);
		queue.enqueueWriteBuffer(inDimBuffer, CL_TRUE, 0, sizeof(int), &imgRows);
		queue.enqueueWriteBuffer(outDimBuffer, CL_TRUE, 0, sizeof(int), &outImgRows);

		std::ifstream sourceFile("cl_kernels/max_pool2d.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "max_pool2d");

     	kernel.setArg(0, channelsBuffer);
     	kernel.setArg(1, inDimBuffer);
     	kernel.setArg(2, poolSizeBuffer);
     	kernel.setArg(3, outDimBuffer);
     	kernel.setArg(4, inputBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(outImgRows, outImgCols);
     	cl::NDRange local(1, 1);
     cl::Event event;
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, channels*outImgRows*outImgCols*sizeof(float), output_buffer);
         cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
// std::cout << "Execution time in milliseconds for maxpool layer " << total_time*1.0e-6f << std::endl;   

	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
 for (int p = 0;p<(channels*outImgRows*outImgCols);p++)
          { 
               
              input_buffer[p] = output_buffer[p]; 
      }
   
      
    }


     else if(layer[j][0]==2)
     {

	/* ------------------------------------ Fully Connected 1 Starts ------------------------------------ */
	
	int in_features, out_features;
	in_features = layer[j][1];
	out_features = layer[j][2];


	//std::cout<<"Performing Fully Connected "<<j<<" "<<std::endl;

	try
	{
		cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*sizeof(float));
		cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_features*sizeof(float));
		cl::Buffer weightsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*out_features*sizeof(float));
		cl::Buffer biasesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_features*sizeof(float));
		cl::Buffer inFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
		cl::Buffer outFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

		queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_features*sizeof(float), input_buffer);
		queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
		queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, in_features*out_features*sizeof(float), w[weight_count]);
		queue.enqueueWriteBuffer(biasesBuffer, CL_TRUE, 0, out_features*sizeof(float), w[weight_count+1]);
		queue.enqueueWriteBuffer(inFeaturesBuffer, CL_TRUE, 0, sizeof(int), &in_features);
		queue.enqueueWriteBuffer(outFeaturesBuffer, CL_TRUE, 0, sizeof(int), &out_features);

		std::ifstream sourceFile("cl_kernels/relu_linear.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

     	cl::Program program = cl::Program(context, source);

     	program.build(devices);
     	
     	cl::Kernel kernel(program, "relu_linear");

     	kernel.setArg(0, inFeaturesBuffer);
     	kernel.setArg(1, outFeaturesBuffer);
     	kernel.setArg(2, inputBuffer);
     	kernel.setArg(3, weightsBuffer);
     	kernel.setArg(4, biasesBuffer);
     	kernel.setArg(5, outputBuffer);

     	cl::NDRange global(out_features, 1);
     	cl::NDRange local(1, 1);
     cl::Event event;
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();

     	queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
               cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
// std::cout << "Execution time in milliseconds for Fully Connected/Dense layer " << total_time*1.0e-6f << std::endl;   


	}
	catch(cl::Error error)
	{
		std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
	}
     for (int n = 0; n < layer[j][2]; n++) {
              input_buffer[n] =  output_buffer[n];
       //       std::cout<<input_buffer[n]<<std::endl;
      }
    weight_count = weight_count+2;  
    } 


 else
     {

  /* ------------------------------------ Fully Connected 1 Starts ------------------------------------ */
  
  int in_features, out_features;
  in_features = layer[j][1];
  out_features = layer[j][2];


  //std::cout<<"Performing Fully Connected "<<j<<" "<<std::endl;

  try
  {
    cl::Buffer inputBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*sizeof(float));
    cl::Buffer outputBuffer = cl::Buffer(context, CL_MEM_WRITE_ONLY, out_features*sizeof(float));
    cl::Buffer weightsBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, in_features*out_features*sizeof(float));
    cl::Buffer biasesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, out_features*sizeof(float));
    cl::Buffer inFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));
    cl::Buffer outFeaturesBuffer = cl::Buffer(context, CL_MEM_READ_ONLY, sizeof(int));

    queue.enqueueWriteBuffer(inputBuffer, CL_TRUE, 0, in_features*sizeof(float), input_buffer);
    queue.enqueueWriteBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
    queue.enqueueWriteBuffer(weightsBuffer, CL_TRUE, 0, in_features*out_features*sizeof(float), w[weight_count]);
    queue.enqueueWriteBuffer(biasesBuffer, CL_TRUE, 0, out_features*sizeof(float), w[weight_count+1]);
    queue.enqueueWriteBuffer(inFeaturesBuffer, CL_TRUE, 0, sizeof(int), &in_features);
    queue.enqueueWriteBuffer(outFeaturesBuffer, CL_TRUE, 0, sizeof(int), &out_features);

    std::ifstream sourceFile("cl_kernels/linear.cl");
      std::string sourceCode(
         std::istreambuf_iterator<char>(sourceFile),
         (std::istreambuf_iterator<char>()));
      cl::Program::Sources source(1,
         std::make_pair(sourceCode.c_str(),
         sourceCode.length() + 1));

      cl::Program program = cl::Program(context, source);

      program.build(devices);
      
      cl::Kernel kernel(program, "linear");

      kernel.setArg(0, inFeaturesBuffer);
      kernel.setArg(1, outFeaturesBuffer);
      kernel.setArg(2, inputBuffer);
      kernel.setArg(3, weightsBuffer);
      kernel.setArg(4, biasesBuffer);
      kernel.setArg(5, outputBuffer);

      cl::NDRange global(out_features, 1);
      cl::NDRange local(1, 1);
     cl::Event event;
      queue.enqueueNDRangeKernel(kernel, cl::NullRange, global, local,NULL,&event);
      queue.finish();

      queue.enqueueReadBuffer(outputBuffer, CL_TRUE, 0, out_features*sizeof(float), output_buffer);
               cl_ulong time_start;
     cl_ulong time_end;
     
     event.wait();
    double total_time;
    event.getProfilingInfo(CL_PROFILING_COMMAND_END, &time_end); 
    event.getProfilingInfo(CL_PROFILING_COMMAND_START, &time_start);
    total_time = time_end - time_start;

/* Results */
// std::cout << "Execution time in milliseconds for Fully Connected/Dense layer " << total_time*1.0e-6f << std::endl;   


  }
  catch(cl::Error error)
  {
    std::cout << error.what() << "(" << error.err() << ")" <<std::endl;
  }
     for (int n = 0; n < layer[j][2]; n++) {
              input_buffer[n] =  output_buffer[n];
        std::cout<<input_buffer[n]<<std::endl;
      }
    weight_count = weight_count+2;  
    } 
}

// exit(0);



  float max = input_buffer[0];
  // std::cout<<input_buffer[3];
      int outputs = 0;
      for (int j = 0; j < 4; j++)
      {

          if (input_buffer[j] >= max)
         {
            outputs = j;
            max = input_buffer[j];
        }
      }

      std::cout<<i<<"   Predicted "<<outputs<<" "<<"Expected" << test_label;
      if ( outputs != test_label )
      {
          std ::cout<<"      "<<"Mismatched";

      }
         else
       {
          
              count = count+1;
        
      }
     std::cout<<" Done"<<std::endl;
    

}
std::cout<<std::endl<<"Accuracy : "<<((count)/testcases)*100; 
	return 0;

}
