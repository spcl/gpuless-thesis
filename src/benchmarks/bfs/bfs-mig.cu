/***********************************************************************************
  Implementing Breadth first search on CUDA using algorithm given in HiPC'07
  paper "Accelerating Large Graph Algorithms on the GPU using CUDA"

  Copyright (c) 2008 International Institute of Information Technology - Hyderabad. 
  All rights reserved.

  Permission to use, copy, modify and distribute this software and its documentation for 
  educational purpose is hereby granted without fee, provided that the above copyright 
  notice and this permission notice appear in all copies of this software and that you do 
  not sell the software.

  THE SOFTWARE IS PROVIDED "AS IS" AND WITHOUT WARRANTY OF ANY KIND,EXPRESS, IMPLIED OR 
  OTHERWISE.

  Created by Pawan Harish.
 ************************************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <cuda.h>


#include <chrono>
#include <vector>
#include <iostream>

#define MAX_THREADS_PER_BLOCK 512

//Structure to hold a node information
struct Node
{
	int starting;
	int no_of_edges;
};


int no_of_nodes;
int edge_list_size;
bool clean_up;
bool IS_COLD;
FILE *fp;

Node* d_graph_nodes;
int* d_graph_edges;
int* h_cost;
Node* h_graph_nodes;
bool *h_graph_mask;
bool *h_updating_graph_mask;
bool *h_graph_visited;
int* h_graph_edges;
int num_of_blocks;
int num_of_threads_per_block;


#include "kernel.cu"
#include "kernel2.cu"

void BFSGraph(int argc, char** argv);

////////////////////////////////////////////////////////////////////////////////
// Main Program
////////////////////////////////////////////////////////////////////////////////
int main( int argc, char** argv) 
{
		// get contexts
		int dev = 0;
		cudaSetDevice(dev);

		struct cudaDeviceProp devProp;
		cudaGetDeviceProperties(&devProp, 0);
		double mem_gb = (double)devProp.totalGlobalMem / (1024 * 1024 * 1024);
		printf("cudaDevAttrMultiProcessorCount(SMs): %d, cudaDevMem:%.2f\n", devProp.multiProcessorCount, mem_gb);
		
		no_of_nodes=0;
		edge_list_size=0;
		clean_up=false;
		IS_COLD=true;
		// program starts
		int count = 101;
		for (int i = 0; i < count; i++)
		{
			auto s = std::chrono::high_resolution_clock::now();
			if (i == count - 1)
			{
				clean_up = true;
			}

			BFSGraph( argc, argv);
			auto e = std::chrono::high_resolution_clock::now();
			auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;
			printf("%.8f\n", d);
			// time is much shorter because of skipping file reading.
		}

}

void Usage(int argc, char**argv){

fprintf(stderr,"Usage: %s <input_file>\n", argv[0]);

}
////////////////////////////////////////////////////////////////////////////////
//Apply BFS on a Graph using CUDA
////////////////////////////////////////////////////////////////////////////////
void BFSGraph( int argc, char** argv) 
{
    char *input_f;
	if(argc!=2){
	Usage(argc, argv);
	exit(0);
	}
	
	input_f = argv[1];
	
	auto s = std::chrono::high_resolution_clock::now();

	int source = 0;

	if(IS_COLD)
	{
		printf("Reading File\n");
		//Read in Graph from a file
		fp = fopen(input_f,"r");
		if(!fp)
		{
			printf("Error Reading graph file\n");
			return;
		}

		fscanf(fp,"%d",&no_of_nodes);

		num_of_blocks = 1;
		num_of_threads_per_block = no_of_nodes;

		//Make execution Parameters according to the number of nodes
		//Distribute threads across multiple Blocks if necessary
		if(no_of_nodes>MAX_THREADS_PER_BLOCK)
		{
			num_of_blocks = (int)ceil(no_of_nodes/(double)MAX_THREADS_PER_BLOCK); 
			num_of_threads_per_block = MAX_THREADS_PER_BLOCK; 
		}

		// allocate host memory
		h_graph_nodes = (Node*) malloc(sizeof(Node)*no_of_nodes);
		h_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
		h_updating_graph_mask = (bool*) malloc(sizeof(bool)*no_of_nodes);
		h_graph_visited = (bool*) malloc(sizeof(bool)*no_of_nodes);
		
		int start, edgeno;   
		// initalize the memory
		for( unsigned int i = 0; i < no_of_nodes; i++) 
		{
			fscanf(fp,"%d %d",&start,&edgeno);
			h_graph_nodes[i].starting = start;
			h_graph_nodes[i].no_of_edges = edgeno;
			h_graph_mask[i]=false;
			h_updating_graph_mask[i]=false;
			h_graph_visited[i]=false;
		}
		//read the source node from the file
		fscanf(fp,"%d",&source);

		fscanf(fp,"%d",&edge_list_size);

		int id,cost;
		h_graph_edges = (int*) malloc(sizeof(int)*edge_list_size);
		for(int i=0; i < edge_list_size ; i++)
		{
			fscanf(fp,"%d",&id);
			fscanf(fp,"%d",&cost);
			h_graph_edges[i] = id;
		}
	

		if(fp)
			fclose(fp);    

		printf("Read File\n");
		
		// start time is override here to skip file reading
		s = std::chrono::high_resolution_clock::now();
		
		cudaMalloc( (void**) &d_graph_nodes, sizeof(Node)*no_of_nodes) ;
		cudaMemcpy( d_graph_nodes, h_graph_nodes, sizeof(Node)*no_of_nodes, cudaMemcpyHostToDevice);

		//Copy the Edge List to device Memory
		
		cudaMalloc( (void**) &d_graph_edges, sizeof(int)*edge_list_size) ;
		cudaMemcpy( d_graph_edges, h_graph_edges, sizeof(int)*edge_list_size, cudaMemcpyHostToDevice);
	
		// allocate mem for the result on host side
		h_cost = (int*) malloc( sizeof(int)*no_of_nodes);
		printf("Copied Everything Static to GPU memory\n");

		IS_COLD = false;
	}

	source=0;

	//set the source node as true in the mask
	h_graph_mask[source]=true;
	h_graph_visited[source]=true;

	//Copy the Mask to device memory
	bool* d_graph_mask;
	cudaMalloc( (void**) &d_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_mask, h_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//Copy the Visited nodes array to device memory
	bool* d_graph_visited;
	cudaMalloc( (void**) &d_graph_visited, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_graph_visited, h_graph_visited, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//reset 
	bool* d_updating_graph_mask;
	cudaMalloc( (void**) &d_updating_graph_mask, sizeof(bool)*no_of_nodes) ;
	cudaMemcpy( d_updating_graph_mask, h_updating_graph_mask, sizeof(bool)*no_of_nodes, cudaMemcpyHostToDevice) ;
	
	for(int i=0;i<no_of_nodes;i++)
		h_cost[i]=-1;
	h_cost[source]=0;

	// allocate device memory for result
	int* d_cost;
	cudaMalloc( (void**) &d_cost, sizeof(int)*no_of_nodes);
	cudaMemcpy( d_cost, h_cost, sizeof(int)*no_of_nodes, cudaMemcpyHostToDevice) ;

	//make a bool to check if the execution is over
	bool *d_over;
	cudaMalloc( (void**) &d_over, sizeof(bool));

	printf("Copied Everything Dynamic to GPU memory\n");

	// setup execution parameters
	dim3  grid( num_of_blocks, 1, 1);
	dim3  threads( num_of_threads_per_block, 1, 1);

	int k=0;
	printf("Start traversing the tree\n");
	bool stop;
	//Call the Kernel untill all the elements of Frontier are not false
	do
	{
		//if no thread changes this value then the loop stops
		stop=false;
		cudaMemcpy( d_over, &stop, sizeof(bool), cudaMemcpyHostToDevice) ;
		Kernel<<< grid, threads, 0 >>>( d_graph_nodes, d_graph_edges, d_graph_mask, d_updating_graph_mask, d_graph_visited, d_cost, no_of_nodes);
		// check if kernel execution generated and error
		

		Kernel2<<< grid, threads, 0 >>>( d_graph_mask, d_updating_graph_mask, d_graph_visited, d_over, no_of_nodes);
		// check if kernel execution generated and error
		

		cudaMemcpy( &stop, d_over, sizeof(bool), cudaMemcpyDeviceToHost) ;
		k++;
	}
	while(stop);


	printf("Kernel Executed %d times\n",k);

	// copy result from device to host
	cudaMemcpy( h_cost, d_cost, sizeof(int)*no_of_nodes, cudaMemcpyDeviceToHost);

    auto e = std::chrono::high_resolution_clock::now();
    auto d = std::chrono::duration_cast<std::chrono::microseconds>(e-s).count() / 1000000.0;
    printf("%.8f\n", d);

	//Store the result into a file
	FILE *fpo = fopen("result.txt","w");
	for(int i=0;i<no_of_nodes;i++)
		fprintf(fpo,"%d) cost:%d\n",i,h_cost[i]);
	fclose(fpo);
	printf("Result stored in result.txt\n");

	// clean up
	h_graph_mask[source]=false;
	h_graph_visited[source]=false;

	if (clean_up)
	{
		// cleanup memory
		free( h_graph_nodes);
		free( h_graph_edges);
		free( h_graph_mask);
		free( h_updating_graph_mask);
		free( h_graph_visited);
		free( h_cost);
		cudaFree(d_graph_nodes);
		cudaFree(d_graph_edges);
		cudaFree(d_graph_mask);
		cudaFree(d_updating_graph_mask);
		cudaFree(d_graph_visited);
		cudaFree(d_cost);
	}
}