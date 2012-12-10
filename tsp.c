/**
* Copyright (c) <2012>, <Victor Mateevitsi> www.vmateevitsi.com
* All rights reserved.
* 
* Redistribution and use in source and binary forms, with or without
* modification, are permitted provided that the following conditions are met:
* 1. Redistributions of source code must retain the above copyright
*    notice, this list of conditions and the following disclaimer.
* 2. Redistributions in binary form must reproduce the above copyright
*    notice, this list of conditions and the following disclaimer in the
*    documentation and/or other materials provided with the distribution.
* 3. All advertising materials mentioning features or use of this software
*    must display the following acknowledgement:
*    This product includes software developed by the <organization>.
* 4. Neither the name of the <organization> nor the
*    names of its contributors may be used to endorse or promote products
*    derived from this software without specific prior written permission.

* THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> ''AS IS'' AND ANY
* EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL <COPYRIGHT HOLDER> BE LIABLE FOR ANY
* DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
* (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
* LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
*	ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
*	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
*	SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*
*/

#include <float.h>
#include <getopt.h>
#include <limits.h>
#include <math.h>
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

int n;
char locations_file[255];
int computerStats = 0;
int local_rank, num_procs;

// Timing
double gen_time, proc_time, comm_time, total_time;

typedef enum { 
  linear,
  nn
} TYPE;
TYPE type;

typedef struct {
  int line;
  float x;
  float y;
  int visited;
} LOCATION;
LOCATION *locations;

typedef struct {
  float cost;
  int pointA;
  int pointB;
} COST;


int parse_arguments(int argc, char **argv);
void parse_file();

void swap(int *p1, int *p2);
void permute();
void nearest_neighbor();
void bubbleSort(COST *array, int array_size);
void display(int *a, int *count, float cost);
float calculate_cost(int *a);
float dist(LOCATION a, LOCATION b);
int find_nearest(int current, int start, int end);

int main(int argc, char **argv) {
	double t_start, t_end;
  int i;

  gen_time = 0.0; proc_time = 0.0; comm_time = 0.0; total_time = 0.0;
 
  // Parse the arguments
  if( parse_arguments(argc, argv) ) return 1;
  parse_file();
  // Initialize MPI
  MPI_Init(&argc, &argv); 
  MPI_Comm_rank(MPI_COMM_WORLD, &local_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

  t_start = MPI_Wtime();
  
  if( type == linear ) {
    permute();
  } else if( type == nn ) {
    nearest_neighbor();
  }

  t_end = MPI_Wtime();
  total_time = t_end - t_start;

  if( computerStats ) {
    printf("%d\tg\t%d\t%d\t%f\n", n, local_rank, num_procs, gen_time);
    printf("%d\tp\t%d\t%d\t%f\n", n, local_rank, num_procs, proc_time);
    printf("%d\tc\t%d\t%d\t%f\n", n, local_rank, num_procs, comm_time);
    printf("%d\tt\t%d\t%d\t%f\n", n, local_rank, num_procs, total_time);
  }

  free(locations);
	MPI_Finalize(); // Exit MPI
	return 0;
}

void display(int *a, int *count, float cost) {   
  int x;
  for( x = 0; x < n; x++ )
    printf("%d  ",a[x]);
  printf("cost:%f count:%d\n", cost, *count);
}

void swap(int *p1, int *p2) {   
  int temp;
  temp = *p1;
  *p1 = *p2;
  *p2 = temp;
}

float dist(LOCATION a, LOCATION b) {
  return sqrt(pow(a.x - b.x, 2) + pow(a.y - b.y, 2));
}

float calculate_cost(int *a) {
  int i;
  float cost = 0.0f;
  for( i = 0; i < n - 1; i++ ) {
    cost += dist(locations[a[i] - 1], locations[a[i + 1] - 1]);
  }

  return cost;
}

void permute() {   
  int count, i, x, y;
  int n_perm = 1;
  int *a;
  float cost = 0.0;;
  double start, end, dt;
 
  start = MPI_Wtime();

  for( i = 1; i <= n; i++ ) {
    n_perm *= i;
  }

  a = (int*) malloc(sizeof(int) * n * n_perm);
  for( i = 0; i < n; i++ ) a[i] = i + 1;

  while( count < n_perm ) {
    for( y = 0; y < n - 1; y++) {  
      swap(&a[y], &a[y + 1]);
      cost = calculate_cost(a);
      if( !computerStats) display(a, &count, cost);
      count++;
    }
    swap(&a[0], &a[1]);
    cost = calculate_cost(a);
    if( !computerStats) display(a, &count, cost);
    count++;
    
    for( y = n - 1; y > 0; y-- ) {  
      swap(&a[y], &a[y - 1]);
      cost = calculate_cost(a);
      if( !computerStats) display(a, &count, cost);
      count++;
    }
    swap(&a[n - 1], &a[n - 2]);
    cost = calculate_cost(a);
    if( !computerStats) display(a, &count, cost);
    count++;
  }
  end = MPI_Wtime();
  dt = end - start;
  proc_time += dt;
}

int find_nearest(int current, int start, int end) {
  int i, index;
  float min = FLT_MAX;
  float distance;

  index = -1;
  for( i = start; i<= end; ++i ) {
    distance = dist(locations[current], locations[i]);
    if( distance < min && i != current && locations[i].visited == 0 ) {
      min = distance;
      index = i;
    }
  }

  return index;
}

void nearest_neighbor() {
  int i, j, index;
  float cost = 0.0f;
  int starting_loc, ending_loc;
  int loc_per_node = n / num_procs;
  int next;
  int *index_of_min;
  float distance;
  float min = FLT_MAX;
  int final_path[n];
  double start, end, dt;


  index_of_min = (int*)malloc(sizeof(int) * num_procs);
  starting_loc = loc_per_node * local_rank;
  ending_loc = starting_loc + loc_per_node - 1;
  if( local_rank == num_procs - 1 ) ending_loc += n % num_procs;
  
  next = 0;
  final_path[0] = 0;
  for( i = 0; i < n - 1; i++ ) {
    // Find the nearest neighbor to
    start = MPI_Wtime();
    MPI_Bcast(&next, 1, MPI_INT, 0, MPI_COMM_WORLD);
    end = MPI_Wtime();
    dt = end - start;
    comm_time += dt;

    start = MPI_Wtime();
    locations[next].visited = 1;
    int index = find_nearest(next, starting_loc, ending_loc);
    end = MPI_Wtime();
    dt = end - start;
    proc_time += dt;

    start = MPI_Wtime();
    MPI_Gather(&index, 1, MPI_INT, index_of_min, 1, MPI_INT, 0, MPI_COMM_WORLD);
    end = MPI_Wtime();
    dt = end - start;
    comm_time += dt;
    
    if( local_rank == 0 ) {
      start = MPI_Wtime();
      index = index_of_min[0];
      // find the nearest
      min = FLT_MAX;
      for( j = 0; j < num_procs; ++j ) {
        if( index_of_min[j] < 0 ) continue;
        distance = dist(locations[next], locations[index_of_min[j]]);
        if( distance < min ) {
          min = distance;
          index = index_of_min[j];
        }
      }
      next = index;
      final_path[i + 1] = index;
      end = MPI_Wtime();
      dt = end - start;
      proc_time += dt;
    }
      
    MPI_Barrier(MPI_COMM_WORLD);
  }
  
  if( local_rank == 0 && !computerStats ) {
    for( i = 0; i < n; ++i ) {
      printf("%d ", final_path[i]);
    }
    printf("\n");
  }
  free(index_of_min);
}

void bubbleSort(COST *array, int array_size) {
  int i, j;
  COST temp;
  
  for( i = (array_size - 1); i > 0; --i ) {
    for( j = 1; j <= i; ++j ) {
      if( array[j-1].cost > array[j].cost ) { 
        temp = array[j-1];
        array[j-1] = array[j];
        array[j] = temp;
      }
    }
  }
}

void parse_file() {
  FILE *fp;
  int i, line;
  char buffer[1024];
  float x, y;

  fp = fopen(locations_file, "r");

  for(i = 0; i < 7; i++) {
    fgets(buffer, 1024, fp);
  }

  while(fscanf(fp, "%d %f %f", &line, &x, &y) > 0 ) {
    if(line == n) break;
    //printf("%d %f %f\n", line, x, y);
  }

  // Line has the number of elements
  locations = (LOCATION *) malloc(sizeof(LOCATION) * line);
  rewind(fp);
  
  for(i = 0; i < 7; i++) {
    fgets(buffer, 1024, fp);
  }

  while(fscanf(fp, "%d %f %f", &line, &x, &y) > 0 ) {
    locations[line - 1].line = line;
    locations[line - 1].x = x;
    locations[line - 1].y = y;
    locations[line - 1].visited = 0;
    
    if(line == n) break;
    //printf("%d %f %f\n", locations[line - 1].line, locations[line - 1].x, locations[line - 1].y);
  }

  fclose(fp);
}

int parse_arguments(int argc, char **argv) {
	int i, c;
  int option_index = 0;
  static struct option long_options[] =
  {
    {"input_method1", required_argument,  0, 'q'},
    {"input_method2", required_argument,  0, 'w'},
    {0, 0, 0, 0}
  };

  char *result = NULL;
  char delims[] = "m";
	while( (c = getopt_long (argc, argv, "f:n:t:c", long_options, &option_index)) != -1 ) {
		switch(c) {
      case 'f':
        strcpy(locations_file, optarg);
        break;
			case 'n':
				n = atoi(optarg);
				break;
      case 'c':
        computerStats = 1;
        break;
      case 't':
        if( strcmp(optarg, "linear" ) == 0 ) type = linear;
        else if( strcmp(optarg, "nn" ) == 0 ) type = nn;
        else {
          fprintf( stderr, "Option -%c %s in incorrect. Allowed values are: linear, nn\n", optopt, optarg);
          return 1;
        }
        break;
      case '?':
				if( optopt == 'n' )
					fprintf (stderr, "Option -%c requires an argument.\n", optopt);
				else if (isprint (optopt))
					fprintf (stderr, "Unknown option `-%c'.\n", optopt);
				else
					fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
				return 1;
			default:
				fprintf(stderr, "Usage: %s -n <number of numbers> \n", argv[0]);
				fprintf(stderr, "\tExample: %s -n 1000\n", argv[0]);
				return 1;
		}
	}
	return 0;
}
