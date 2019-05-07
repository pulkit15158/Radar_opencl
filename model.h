#ifndef MODEL_H_
#define MODEL_H_

#define FILTER_SIZe 9
#define NUMBER_LAYERS 9
#define NUMBER_WEIGHTS 10

#define OUTPUT_BUFFER_SIZE 10000000
#define INPUT_BUFFER_SIZE 10000000


int  imgRows =  128;
int  imgCols = 384;
int  imgChannels = 1;
unsigned layer[NUMBER_LAYERS][8] = {
						
						{0,1,32,130,386,128,384,3},
						
						{1,32,32,128,384,64,192,2},
						
						{0,32,32,66,194,64,192,3},
						{1,32,32,64,192,32,96,2},
						
						{0,32,64,34,98,32,96,3},
						
						{1,64,64,32,96,16,48,2},
						{1,64,64,16,48,8,24,2},
							
						{2,12288,256,0,0,0},
						{3,256,4,0,0,0},
						
						};

int weights[] = {288,32,9216,32,18432,64,3145728,256,1024,4};  
int weights_size[] = {0,288,320,9536,9568,28000,28064,3173792,3174048,3175072,3175076};
#endif
