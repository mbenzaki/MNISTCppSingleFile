#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <sstream>
#include <fstream>
#include <tuple>
#include <initializer_list>
#include <algorithm>
#include <memory>


using namespace std;


struct Matrix
{

	///// Must be private for object oriented approach:
	vector<vector<float>>	_data;
	size_t					_row;
	size_t					_column;

public:

	//
	// Default Constructor
	//
	Matrix() {};

	//
	// Copy constructor
	//
	Matrix(const Matrix &rhs)
		:Matrix(rhs.getRow(), rhs.getColumn())
	{
		for (size_t i = 0; i < rhs.getRow(); ++i)
		{
			for (size_t j = 0; j < rhs.getColumn(); ++j)
			{
				_data[i][j] = rhs._data[i][j];
			}
		}

	};

	//
	// Constructor with number of row and column and not initialized
	//
	Matrix(const size_t row, const size_t column)
		:_row(row)
		, _column(column)
	{
		_data.resize(row);
		for (size_t i = 0; i < row; ++i)
		{
			_data[i].resize(column);
		}
	};


	//
	// Constructor with number of row and column and initial value
	//
	Matrix(const size_t row, const size_t column, const float epsilon)
		:_row(row)
		, _column(column)
	{
		random_device rnd;

		_data.resize(row);
		for (size_t i = 0; i < row; ++i)
		{
			_data[i].resize(column);
			for (size_t j = 0; j < column; ++j)
			{
				_data[i][j] = static_cast<float>((2.0 * rnd() - 0xffffffff) / 0xffffffff * epsilon);
			}
		}
	};

	//
	// Constructor with number of row and column and initial value of by one dim.
	//
	Matrix(const vector<float> & vec)
		:Matrix(1, vec.size())
	{
		for (size_t i = 0; i < _column; ++i)
		{
			_data[0][i] = vec[i];
		}
	};

	//
	// Constructor with number of row and column and initial value of by two dim.
	//
	Matrix(const vector<vector<float>> & mat)
		:Matrix(mat.size(), mat[0].size())
	{
		for (size_t i = 0; i < _row; ++i)
		{
			if (mat[i].size() != _column)
			{
				throw("Contructor with vec<vec<float>> initializer, it is NOT matrix");
			}
			for (size_t j = 0; j < _column; ++j)
			{
				_data[i][j] = mat[i][j];
			}
		}
	};

	//
	// Getter and Setter, in this code, you can access member value without
	// getter and setter because all menmbers are public
	//
	float getData(size_t i, size_t j) const { return _data[i][j]; };
	void setData(size_t i, size_t j, float value) { _data[i][j] = value; };
	size_t getRow()const { return _row; };
	size_t getColumn()const { return _column; };

	//
	// Set up matrix with number of row and column and not initialized
	//
	void setup(const size_t row, const size_t column)
	{
		_row = row;
		_column = column;

		_data.resize(row);
		for (size_t i = 0; i < row; ++i)
		{
			_data[i].resize(column);
		}
	};


	//
	// Set up matrix with number of row and column and initial value
	//
	void setup(const size_t row, const size_t column, const float epsilon)
	{
		random_device rnd;

		_row = row;
		_column = column;

		_data.resize(row);
		for (size_t i = 0; i < row; ++i)
		{
			_data[i].resize(column);
			for (size_t j = 0; j < column; ++j)
			{
				_data[i][j] = static_cast<float>((2.0 * rnd() - 0xffffffff) / 0xffffffff * epsilon);
			}
		}
	};

	//
	// Return row and column as tuple
	//
	tuple<size_t, size_t> shape() const
	{
		return std::make_tuple(_row, _column);
	};

	//
	// Calculate summation of all value in data
	//
	float sum() const
	{
		float result = 0.0;
		for (size_t i = 0; i < _row; ++i)
		{
			for (size_t j = 0; j < _column; ++j)
			{
				result += _data[i][j];
			}
		}
		return result;
	};

	//
	// Calculate summation of all value in data
	//
	float max() const
	{
		float result = -FLT_MAX;
		for (size_t i = 0; i < _row; ++i)
		{
			result = std::max(result, *max_element(_data[i].begin(), _data[i].end()));
		}
		return result;
	};


	//
	// Calculate sigmoid
	//
	void sigmoid()
	{
		for (size_t i = 0; i < _row; ++i)
		{
			for (size_t j = 0; j < _column; ++j)
			{
				_data[i][j] = static_cast<float>(1.0 / (1.0 + exp(-_data[i][j])));
			}
		}
	}

	//
	// Calculate relu
	//
	void relu()
	{
		for (size_t i = 0; i < _row; ++i)
		{
			for (size_t j = 0; j < _column; ++j)
			{
				_data[i][j] = std::max(0.0f, _data[i][j]);
			}
		}
	}

	//
	// Slice of Matrix
	//
	Matrix slice(size_t start, size_t rows) const
	{
		Matrix result(rows, this->getColumn());
		for (size_t i = 0; i < rows; ++i)
		{
			for (size_t j = 0; j < _column; ++j)
			{
				result._data[i][j] = this->_data[start + i][j];
			}
		}
		return result;
	}

	//
	// Return the index of max value for each row as vector
	//
	vector<uint16_t> maxIndex() const
	{
		vector<uint16_t> result;
		result.resize(_row);

		for (auto i = 0; i < _row; ++i)
		{
			result[i] = 0;
			auto maxValue = _data[i][0];
			for (auto j = 1; j < _column; ++j)
			{
				if (maxValue < _data[i][j])
				{
					maxValue = _data[i][j];
					result[i] = j;
				}
			}
		}
		return result;
	}

	// *****************************
	// static functions
	// *****************************

	//
	// Return translate matrix 
	//
	static Matrix trans(const Matrix & matrix)
	{
		Matrix result(matrix._column, matrix._row);

		for (size_t i = 0; i < matrix._row; ++i)
		{
			for (size_t j = 0; j < matrix._column; ++j)
			{
				result._data[j][i] = matrix._data[i][j];
			}

		}
		return result;
	}

	//
	// Return inner product of two matrixes
	//
	static Matrix dot(const Matrix & lhs, const Matrix & rhs)
	{
		if (lhs._column != rhs._row)
		{
			throw new exception("Row of rhs and coloum of rhs mismatch");
		}

		size_t inner = lhs._column;
		size_t row = lhs._row;
		size_t column = rhs._column;

		// Initialize 0
		Matrix result(row, column, 0.0);

		for (size_t i = 0; i < row; ++i)
		{
			for (size_t j = 0; j < column; ++j)
			{
				for (size_t k = 0; k < inner; ++k)
				{
					result._data[i][j] += lhs._data[i][k] * rhs._data[k][j];
				}
			}
		}

		return result;
	}

	//
	// Calculate sigmoid
	//
	static Matrix sigmoid(const Matrix & matrix)
	{
		Matrix result(matrix._row, matrix._column);
		for (size_t i = 0; i < matrix._row; ++i)
		{
			for (size_t j = 0; j < matrix._column; ++j)
			{
				result._data[i][j] = static_cast<float>(1.0 / (1.0 + exp(-matrix._data[i][j])));
			}
		}
		return result;
	}

	//
	// Calculate relu
	//
	static Matrix relu(const Matrix & matrix)
	{
		Matrix result(matrix._row, matrix._column);
		for (size_t i = 0; i < matrix._row; ++i)
		{
			for (size_t j = 0; j < matrix._column; ++j)
			{
				result._data[i][j] = std::max(0.0f, matrix._data[i][j]);
			}
		}
		return result;
	}

	//
	// Calculate softmax of all value in data
	//
	static Matrix softmax(const Matrix & matrix)
	{
		Matrix result(matrix._row, matrix._column);
		double fMax = matrix.max();

		for (size_t i = 0; i < matrix._row; ++i)
		{
			double sumExp = 0.;
			for (size_t j = 0; j < matrix._column; ++j)
			{
				double exponemt = exp(matrix._data[i][j] - fMax);
				sumExp += exponemt;
				result._data[i][j] = static_cast<float>(exponemt);
			}
			for (size_t j = 0; j < matrix._column; ++j)
			{
				result._data[i][j] = static_cast<float>(result._data[i][j] / sumExp);
			}

		}

		return result;
	};
};


//
// Output contents of tuple of shape (row and column)
//
std::ostream& operator<<(std::ostream& os, const tuple<size_t, size_t> & shape)
{
	size_t row, column;

	std::tie(row, column) = shape;
	os << "(row=" << row << ", column=" << column << ")";

	return os;
}

//
// Output contents of matrix
/*
np.array ([
[0.779659, -0.755285, 0.225202, 0.326331],
[0.129933, -0.653668, -0.540027, 0.449548],
])
*/
std::ostream& operator<<(std::ostream& os, const Matrix& matrix)
{
	tuple<size_t, size_t> tpl = matrix.shape();
	os << "np.array ([" << endl;
	for (size_t i = 0; i < matrix.getRow(); ++i)
	{
		os << "  [";
		for (size_t j = 0; j < matrix.getColumn(); ++j)
		{
			os << " " << fixed << matrix.getData(i, j) << ",";
		}
		os << " ]," << endl;
	}
	os << "])" << endl;
	os << "  Shape " << matrix.shape() << endl;

	return os;
}

class Network
{

protected:
	Matrix	_x;		// Input Layer
	Matrix	_y;		// Output Layer

public:

	Network()
		: _setup(false)
	{
	}

	virtual ~Network()
	{
	}

	//
	// Setup Network
	//
	virtual void setup() = 0;

	//
	// Predict value
	//
	virtual double predict(size_t batchSize = 0)
	{
		throw exception("Not Implemented yet");
		return DBL_MAX;
	};

	//
	// Forward Propagation
	//  return value is y which is matrix calculated by this neural network
	virtual void forward(size_t batchSize = 1)
	{
		throw exception("Not Implemented yet");
	};

	//
	// Backward Propagation
	//
	virtual void backword()
	{
		throw exception("Not Implemented yet");
	};

protected:
	bool	_setup;
};

class ThreeLayerNeuralNetwork : public Network
{

protected:
	static const size_t	MninstImageSize = (28 * 28);

	size_t _numOfItems;	// Total number of Supervised data
	Matrix _x;	// Input Layer Matrix (_numItems x (MnistImageSize))
	Matrix _t;	// Supervised data from MNIST

	Matrix _w1;	// Weight from input layer to hidden layer(a1)		shape(28x28, 50)
	Matrix _w2;	// Weight from hidden layer(a1) to hidden layer(a2) shape(50,100)
	Matrix _w3;	// Weight from hidden layer(a2) output layer		shape(100,10)

	size_t	_numOfTotalCalculated;
	size_t	_numOfaccuracy;


	static int32_t readInt32(istream & is)
	{

		// MNIST is big/high endian
		int32_t number;
		unsigned char data[4];

		number = 0;
		is.read(reinterpret_cast<char*>(data), 4);
		number += data[0] << 24;
		number += data[1] << 16;
		number += data[2] << 8;
		number += data[3];

		return number;
	}


	void loadMNISTData()
	{
		// read images of traing set
		const streamsize bufferSize = 0x10000;
		const string imageFileName = "..\\Data\\train-images-idx3-ubyte";
		const string labelFileName = "..\\Data\\train-labels-idx1-ubyte";
		stringstream ss;

		int32_t magic;

		{
			const char * fileName = imageFileName.c_str();

			// Read MNIST Training Set Image
			ifstream ifs(imageFileName, fstream::in | fstream::binary);
			//	ifs.rdbuf()->pubsetbuf(0, bufferSize);
			if (!ifs)
			{
				ss << "File(" << fileName << ") can not be opened";
				throw exception(ss.str().c_str());
			}

			magic = readInt32(ifs);
			if (magic != 0x0803)
			{
				ss << "File(" << fileName << ") magic number is wrong, it might not MNIST image file";
				throw exception(ss.str().c_str());
			}
			_numOfItems = readInt32(ifs);
			size_t rows = readInt32(ifs);
			size_t columns = readInt32(ifs);
			if ((rows != 28) || (columns != 28))
			{
				ss << "File(" << fileName << ") rows or/and columns is not 28";
				throw exception(ss.str().c_str());
			}

			vector<uint8_t> buffer(MninstImageSize * _numOfItems);
			ifs.read(reinterpret_cast<char*>(&buffer[0]), MninstImageSize*_numOfItems);
			if (!ifs)
			{
				ss << "error: only " << ifs.gcount() << " could be read";
				throw exception(ss.str().c_str());
			}
			ifs.close();

			// resize the matrix of input layer x
			_x.setup(_numOfItems, MninstImageSize);
			for (size_t i = 0; i < _numOfItems; ++i)
			{
				for (size_t j = 0; j < MninstImageSize; ++j)
				{
					// Normarize the input value
					_x._data[i][j] = buffer[i*MninstImageSize + j] / 255.0f;
				}
			}
			// End of set up MNIST image file
		}

		{
			// Start of set up MINIS label file
			const char * fileName = labelFileName.c_str();

			// Read MNIST Training Set Label
			ifstream ifs(labelFileName, fstream::in | fstream::binary);

			if (!ifs)
			{
				ss << "File(" << fileName << ") can not be opened";
				throw exception(ss.str().c_str());
			}

			magic = readInt32(ifs);
			if (magic != 0x0801)
			{
				ss << "File(" << fileName << ") magic number is wrong, it might not MNIST label file";
				throw exception(ss.str().c_str());
			}
			size_t numOfItems = readInt32(ifs);
			if (numOfItems != _numOfItems)
			{
				ss << "A number of items is mismatched Image File(" << imageFileName.c_str() << ") Label File("
					<< fileName << ")";
				throw exception(ss.str().c_str());
			}

			vector<uint8_t> buffer(_numOfItems);
			ifs.read(reinterpret_cast<char*>(&buffer[0]), _numOfItems);
			if (!ifs)
			{
				ss << "error: only " << ifs.gcount() << " could be read";
				throw exception(ss.str().c_str());
			}
			ifs.close();

			// resize the matrix of input layer x
			_t.setup(_numOfItems, 10, 0.0);
			for (size_t i = 0; i < _numOfItems; ++i)
			{
				_t._data[i][buffer[i]] = 1.0f;
			}
			ifs.close();
		}
	}

public:
	ThreeLayerNeuralNetwork() {};
	virtual ~ThreeLayerNeuralNetwork() {};
	virtual void setup()
	{

		//
		// Load MNIST data set(image and label)
		//
		loadMNISTData();

		//
		// Set up weights of input, a1, a2, a3 and output
		//

		// Weight from input layer to hidden layer(a1)		shape(28x28, 50)
		_w1.setup(MninstImageSize, 50, 1.0);

		// Weight from hidden layer(a1) to hidden layer(a2) shape(50,100)
		_w2.setup(50, 100, 1.0);

		// Weight from hidden layer(a2) output layer		shape(100,10)
		_w3.setup(100, 10, 1.0);

	}

	virtual void forward(size_t batchSize = 1)
	{

		//
		// Initialize  values of accuracy calculation 
		//
		_numOfTotalCalculated = 0;
		_numOfaccuracy = 0;

		auto totalRows = _t.getRow();
		int64_t miniBatchSize;

		for (size_t i = 0; ; i += batchSize)
		{
			if (i + batchSize > totalRows)
			{
				miniBatchSize = totalRows - i;
				if (miniBatchSize < 1)	break;
			}
			else
			{
				miniBatchSize = static_cast<size_t>(batchSize);
			}

			auto subX = _x.slice(i, miniBatchSize);
			cout << "i=" << i << ", miniBatchSize=" << miniBatchSize <<
				" shape=" << subX.shape() << endl;
			auto result = forwardMiniBatch(subX, false);

			//
			// Calculate accuracy
			//

			// Slice of teacher data
			auto subT = _t.slice(i, miniBatchSize);
			if (subX.getRow() != subT.getRow())
			{
				throw exception("something wrong 1. in forward");
			}

			auto resultMaxIndex = result.maxIndex();
			auto tMaxIndex = subT.maxIndex();
			if (resultMaxIndex.size() != tMaxIndex.size())
			{
				throw exception("something wrong 2. in forward");
			}

			for (auto i = 0; i < resultMaxIndex.size(); ++i)
			{
				if (resultMaxIndex[i] == tMaxIndex[i])	_numOfaccuracy++;
			}

			_numOfTotalCalculated += resultMaxIndex.size();

			cout << "*** Current Accuracy is " << fixed
				<< _numOfaccuracy << " of " << _numOfTotalCalculated << " "
				<< static_cast<double>(_numOfaccuracy) / _numOfTotalCalculated * 100.0
				<< "%" << endl << endl;
		}
	}

	//
	// Forward Propagation
	//  return value is y which is matrix calculated by this neural network
	virtual Matrix forwardMiniBatch(const Matrix & subX, bool bSoftmax = false)
	{
		Matrix y;

		{
			// Ignore batch size, try toi all data
			//   MninstImageSize is 28 x 28 = 784

			// a1's shape is (miniBatchSize, 50)
			Matrix a1 = Matrix::dot(subX, _w1);
			Matrix::sigmoid(a1);

			// a2's shape is (miniBatchSize, 100)
			Matrix a2 = Matrix::dot(a1, _w2);
			Matrix::sigmoid(a2);

			// a3 and y's shape is (miniBatchSize, 100)
			Matrix a3 = Matrix::dot(a2, _w3);

			if (bSoftmax)
			{
				y = Matrix::softmax(a3);
			}
			else
			{
				y = a3;
			}
		}

		return y;
	};

};


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

