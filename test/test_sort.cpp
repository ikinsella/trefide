#include <algorithm>
#include <iostream>
#include <cstdlib>

using namespace std;

struct MyComparator
{
    const double* value_vector;

    MyComparator(const double* val_vec):
        value_vector(val_vec) {}

    bool operator()(int i1, int i2)
    {
        return value_vector[i1] > value_vector[i2];
    }
};

void print(const int size, const double* v, const char * msg)
{
    for (int i = 0; i < size; ++i)
        cout << v[i] << " ";

    cout << msg << endl;
}


int main()
{
    srand(time(0));

    double *A, *B;
    A = (double *) malloc(5 * sizeof(double));
    B = (double *) malloc(5 * sizeof(double));

    for (int i = 0; i < 5; ++i)
    {
        A[i] = rand() % 10;
        B[i] = i;
    }

    print(5, A, "<- A");
    print(5, B, "<- B");

    sort(B, B+5, MyComparator(A));

    print(5,B, "<- B (sorted)");

    cout << "\n(hit enter to quit)";
    cin.get();

    return 0;
}
