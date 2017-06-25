#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <algorithm>
#include <memory>

#include "Matrix.h"
#include "Network.h"

using namespace std;


void testConstructor()
{
	cout << "****************Test " << __func__ << "****************" << endl;;
	Matrix  w(3, 4, 1.0);
	cout << "w = " << w;
	cout << "np.max(w) is " << w.max() << endl;
	cout << "np.sum(w) is " << w.sum() << endl;
	cout << "np.shape(w) is " << w.shape() << endl;
	cout << endl;

	Matrix a({
		{ 1., 2., 3., 4. },
		{ 5., 6., 7., 8. },
		{ 9., 10.,11., 12. }
	});
	cout << "a = " << a;
	cout << "np.max(a) is " << a.max() << endl;
	cout << "np.sum(a) is " << a.sum() << endl;
	cout << endl;

	Matrix b({
		{ -1, -2, -3 },
		{ -4, -5, -6 }
	});
	cout << "b = " << b;
	cout << "np.max(b) is " << b.max() << endl;
	cout << "np.sum(b) is " << b.sum() << endl;
	cout << endl;

	vector<vector<float>> cdata = {
		{ 1.0f },
		{ 2.0f },
		{ 3.0f },
	};
	Matrix c(cdata);
	cout << "c = " << c;
	cout << "np.max(c) is " << c.max() << endl;
	cout << "np.sum(c) is " << c.sum() << endl;
	cout << endl;
}

void testDot()
{
	cout << "****************Test " << __func__ << "****************" << endl;;
	Matrix a({
		{ 1., 2., 3., 4. },
		{ 5., 6., 7., 8. },
		{ 9., 10.,11., 12. }
	});
	cout << "a = " << a << endl;

	Matrix t = Matrix::trans(a);
	cout << "a.T is " << t << endl;
		
	cout << "np.dot(a,a.T) is " << Matrix::dot(a, Matrix::trans(a));
	cout << endl;
}

void testSigmoid()
{
	cout << "****************Test " << __func__ << "****************" << endl;;
	vector<float> data = { -1,0, 1.0f, 2.0f };
	Matrix a(data);
	cout << "a = " << a << endl;

	Matrix sig = Matrix::sigmoid(a);
	cout << "Class Method = " << sig << endl;

	a.sigmoid();
	cout << "Member Method is " << a << endl;
	cout << endl;
}

void testMaxIndex()
{
	cout << "****************Test " << __func__ << "****************" << endl;;
	Matrix a({
		{ 1., 2., 3., 4. },
		{ 5., 6., 7., 8. },
		{ 9., 10.,11., 12. }
	});

	auto vec = a.maxIndex();
	cout << "a = " << a ;

	cout << "Max Index= [ " << endl;
	for (auto num : vec) cout << "  " << num << endl;
	cout << "]" << endl << endl;

	Matrix b(3, 20, 10.0);
	vec = b.maxIndex();
	cout << "b = " << b;
	cout << "Max Index= [ " << endl;
	for (auto num : vec) cout << "  " << num << endl;
	cout << "]" << endl << endl;

	Matrix c(10, 5, 30.0);
	vec = c.maxIndex();
	cout << "c = " << c;
	cout << "Max Index= [ " << endl;
	for (auto num : vec) cout << "  " << num << endl;
	cout << "]" << endl << endl;

}

void testSoftmax()
{
	cout << "****************Test " << __func__ << "****************" << endl;;
	vector<float> data = { 0.3f, 2.9f, 4.0f };
	Matrix a(data);
	cout << "a = " << a << endl;

	Matrix softmax = Matrix::softmax(a);
	cout << "softmax of a = " << softmax << endl;

	cout << endl;
}

int main()
{

	try
	{

#if 1
		auto_ptr<Network>	network1(new ThreeLayerNeuralNetwork());
		network1->setup();
		network1->forward(100); // mini batch size is 100
#elif
		// Test code
		testConstructor();
		testDot();
		testSigmoid();
		testMaxIndex();
		testSoftmax();
#endif
	}
	catch (exception & e)
	{
		cerr << "[ERR] Exception occured " << e.what() << endl;
	}

	cout << "Wait to terminate: Push enter key>";
	getchar();
	return 0;
}

