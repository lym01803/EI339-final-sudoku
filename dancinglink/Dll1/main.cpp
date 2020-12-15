#include "pch.h"
#include <iostream>
using namespace std;
struct node
{
	node *left, *right, *up, *down;
	bool ishead;
	int col;
	int id;
	node(bool ish = 0, int co = 0, node *l = nullptr, node *r = nullptr, node *u = nullptr, node *d = nullptr)
	{
		left = l;
		right = r;
		up = u;
		down = d;
		ishead = ish;
		col = co;
	}
};
node *column[330];
node *Head;
bool Row[10][10], Col[10][10], Area[10][10];
int ban[1200];
void Cancel(int col)
{
	++ban[col];
	if(ban[col] == 1)
	{
		node *p = column[col];
		p->left->right = p->right;
		p->right->left = p->left;
	}
	return ;
}
void Recover(int col)
{
	--ban[col];
	if(ban[col] == 0)
	{
		node *p = column[col];
		p->left->right = p;
		p->right->left = p;
	}
	return ;
}
int Ans[2000];
int anstot = 0;
node* delQue[2000];
int deltot = 0;
bool Solve()
{
	if(Head->left == Head)
	{
		return true;
	}
	int delst = deltot;
	for(node* ptr = Head->left->down; !ptr->ishead; ptr = ptr->down)
	{
		node *pptr = ptr;
		delQue[++deltot] = ptr;
		for(int i= 0; i ^ 4; ++i)
		{
			pptr->up->down = pptr->down;
			pptr->down->up = pptr->up;
			pptr = pptr->left;
		}
	}
	for(int i = delst + 1, flag; i <= deltot; ++i)
	{
		node *pptr = delQue[i];
		flag = 1;
		for(int i = 0; i ^ 4; ++i)
		{
			if(ban[pptr->col])
			{
				flag = 0;
			}
			Cancel(pptr->col);
			pptr = pptr->left;
		}
		Ans[++anstot] = pptr->id;
		if(flag && Solve())
		{
			return true;
		}
		--anstot;
		pptr = pptr->right;
		for(int i = 0; i ^ 4; ++i)
		{
			Recover(pptr->col);
			pptr = pptr->right;
		}
	}
	for(; deltot > delst;)
	{
		node *pptr = delQue[deltot--];
		for(int i = 0; i ^ 4; ++i)
		{
			pptr->up->down = pptr;
			pptr->down->up = pptr;
			pptr = pptr->left;
		}
	}
	return false;
}

extern "C" _declspec(dllexport) int __cdecl solve(int[][10]);

int solve(int A[][10])
{
	Head = new node(1);
	Head->left = Head->right = Head;
	node *tmp = Head;
	for(int i = 0; i ^ 324; ++i)
	{
		node *nd = new node(1, i, tmp, tmp->right);
		nd->up = nd->down = nd;
		tmp->right->left = nd;
		tmp->right = nd;
		tmp = nd;
		column[i] = nd;
	}
	for(int i = 0; i ^ 9; ++i)
	{
		for(int j = 0, k; j ^ 9; ++j)
		{
			//cin >> A[i][j];
			if(k = A[i][j])
			{
				--k;
				Row[i][k] = 1;
				Col[j][k] = 1;
				Area[(i/3)*3+j/3][k] = 1;
				int col = (i << 3) + i + j;
				Cancel(col); //(i,j)
				col = 81 + (i << 3) + i + k;
				Cancel(col); //i k
				col = 162 + (j << 3) + j + k;
				Cancel(col); //j k
				col = 243 + ((i/3) * 3 + (j/3)) * 9 + k;
				Cancel(col);
			}
		}
	}
	for(int i = 0; i ^ 9; ++i)
	{
		for(int j = 0; j ^ 9; ++j)
		{
			if(!A[i][j])
			{
				for(int k = 0; k ^ 9; ++k)
				{
					if(!Row[i][k] && !Col[j][k] && !Area[(i/3)*3+j/3][k])
					{
						node *n1, *n2, *n3, *n4; //4 node per row

						int col = (i << 3) + i + j;
						n1 = new node(0, col, nullptr, nullptr, column[col], column[col]->down);
						n1->id = ((i * 9) + j) * 9 + k;
						column[col]->down->up = n1;
						column[col]->down = n1; 
						column[col] = n1; //(i,j)

						col = 81 + (i << 3) + i + k;
						n2 = new node(0, col, nullptr, nullptr, column[col], column[col]->down);
						n2->id = n1->id;
						column[col]->down->up = n2;
						column[col]->down = n2; 
						column[col] = n2;//row, k

						col = 162 + (j << 3) + j + k;
						n3 = new node(0, col, nullptr, nullptr, column[col], column[col]->down);
						n3->id = n2->id;
						column[col]->down->up = n3;
						column[col]->down = n3; 
						column[col] = n3;//column, k

						col = 243 + ((i/3) * 3 + (j/3)) * 9 + k;
						n4 = new node(0, col, nullptr, nullptr, column[col], column[col]->down);
						n4->id = n3->id;
						column[col]->down->up = n4;
						column[col]->down = n4; 
						column[col] = n4;//3*3, k

						n1->left = n4;
						n1->right = n2;
						n2->left = n1;
						n2->right = n3;
						n3->left = n2;
						n3->right = n4;
						n4->left = n3;
						n4->right = n1;
					}
				}
			}
		}
	}
	for(int i = 0; i ^ 324; ++i)
	{
		while(!column[i]->ishead)
		{
			column[i] = column[i]->down;
		}
	}
	//cout << endl;
	if(Solve())
	{
		for(int i = 1; i <= anstot; ++i)
		{
			int p = Ans[i];
			A[p/81][p/9%9] = p%9 + 1;
		}
		/*
		for(int i = 0; i ^ 9; ++i)
		{
			for(int j = 0; j ^ 9; ++j)
			{
				cout << A[i][j] << ' ';
			}
			cout << endl;
		}
		*/
	}
	else
	{
		return -1;
	}
	return 0;
}


/*

0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0
0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 
0 0 0 0 0 0 0 0 0 

0 5 4 9 3 8 2 6 1
1 2 8 6 4 5 9 3 7
6 3 9 2 1 7 4 8 5
8 6 5 4 2 9 1 7 3
9 7 2 3 5 1 6 4 8
4 1 3 8 7 6 5 2 9
5 4 7 1 8 2 3 9 6
2 9 1 7 6 3 8 5 4
3 8 6 5 9 4 7 0 0

0 5 4 9 3 8 2 6 1
1 2 8 6 4 5 9 3 7
6 3 9 2 1 7 4 8 5
8 6 5 4 2 9 1 7 3
9 7 2 3 5 1 6 4 8
4 1 3 8 7 6 5 2 9
5 4 7 1 8 2 3 9 6
2 9 1 7 6 0 0 0 0
0 0 0 0 0 0 0 0 0


7 0 4 9 0 8 2 6 1
1 0 8 0 4 5 9 0 7
6 3 0 2 0 7 0 8 5
8 6 5 4 2 9 1 7 0
0 7 0 3 0 1 0 4 0
4 1 3 8 7 6 5 2 9
0 4 0 1 0 2 0 9 6
2 0 1 7 6 3 8 0 4
3 8 6 5 0 4 7 1 2
*/