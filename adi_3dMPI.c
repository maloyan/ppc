#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N   (2*2*2*2*2*2+2)
double   maxeps = 0.1e-7;
int itmax = 100;
int i,j,k;

double realeps;
double eps;
double sum;
double *A;
double *B;
int rank, numtasks;
int startrow, lastrow, nrow;
int ll, shift;
MPI_Request req[4];
MPI_Status status[4];

void relax();
void init();
void verify();

static int indexNum(int i, int j, int k)
{
	return N*N*i + N*j + k;
}


int main(int argc, char **argv)
{
	MPI_Init(&argc, &argv);
	double start = MPI_Wtime();
	MPI_Comm_size(MPI_COMM_WORLD, &numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	startrow = (rank * N) / numtasks;
	lastrow = ((rank + 1) * N) / numtasks - 1;
	nrow = lastrow - startrow + 1;
	
	A = (double *)calloc((nrow+2)*N*N, sizeof(double));
	B = (double *)calloc(nrow*N*N, sizeof(double));
	
	int it;

	init();
	
	for(it=1; it<=itmax; it++)
	{
		relax();
	}
	
	verify();
	double finish = MPI_Wtime();
	MPI_Finalize();
	printf("time = %lf\n", finish - start);
	return 0;
}


void init()
{
	for(i=1; i<=nrow; i++)
	for(j=0; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
	{
		A[indexNum(i,j,k)] = 0;
		B[indexNum(i-1,j,k)]= ( 4. + startrow + i - 1 + j + k) ;
	}
}

void relax()
{
	for(i=1; i<=nrow; i++){
		if(((i==1)&&(rank==0))||((i==nrow)&&(rank==numtasks-1)))
			continue;
		for(j=1; j<=N-2; j++)
		for(k=1; k<=N-2; k++)
			A[indexNum(i,j,k)] = B[indexNum(i-1,j,k)];
	}
	if(rank != 0)
		MPI_Irecv(&A[indexNum(0,0,0)], N*N, MPI_DOUBLE, rank-1, 1215, MPI_COMM_WORLD, &req[0]);
	if(rank != numtasks - 1)
		MPI_Isend(&A[indexNum(nrow,0,0)], N*N, MPI_DOUBLE, rank+1, 1215, MPI_COMM_WORLD, &req[2]);
	if(rank != numtasks - 1)
		MPI_Irecv(&A[indexNum(nrow+1,0,0)], N*N, MPI_DOUBLE, rank+1, 1216, MPI_COMM_WORLD, &req[3]);
	if(rank != 0)
		MPI_Isend(&A[indexNum(1,0,0)], N*N, MPI_DOUBLE, rank-1, 1216, MPI_COMM_WORLD, &req[1]);
	
	ll = 4; shift = 0;
	if(rank == 0){
		ll=2;
		shift=2;
	}
	if(rank==numtasks-1)
		ll=2;
	if(numtasks != 1) 
		MPI_Waitall(ll, &req[shift], &status[0]);
	
	for(i=1; i<=nrow; i++){
		if(((i==1)&&(rank==0))||((i==nrow)&&(rank==numtasks-1)))
			continue;
		for(j=1; j<=N-2; j++)
		for(k=1; k<=N-2; k++)
			B[indexNum(i-1,j,k)] = (A[indexNum(i-1,j,k)]+A[indexNum(i+1,j,k)])/2.;
	}
	for(i=1; i<=nrow; i++){
		if(((i==1)&&(rank==0))||((i==nrow)&&(rank==numtasks-1)))
			continue;
		for(j=1; j<=N-2; j++)
		for(k=1; k<=N-2; k++)
			B[indexNum(i-1,j,k)] =(B[indexNum(i-1,j-1,k)]+B[indexNum(i-1,j+1,k)])/2.;
	}
	
	for(i=1; i<=nrow; i++){
		if(((i==1)&&(rank==0))||((i==nrow)&&(rank==numtasks-1)))
			continue;
		for(j=1; j<=N-2; j++)
		for(k=1; k<=N-2; k++)
			B[indexNum(i-1,j,k)] = (B[indexNum(i-1,j,k-1)]+B[indexNum(i-1,j,k+1)])/2.;
	}
	
}

void verify()
{
	double s=0.;
	for(i=1; i<=nrow; i++)
	for(j=1; j<=N-1; j++)
	for(k=0; k<=N-1; k++)
		s=s+B[indexNum(i-1,j,k)]*(i -1 + startrow)*(j+1)*(k+1)/(N*N*N);
	MPI_Reduce(&s, &sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	if(rank == 0)
		printf("  S = %f\n",sum);
}