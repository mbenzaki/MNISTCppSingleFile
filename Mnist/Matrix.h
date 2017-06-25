#pragma once

#include <iostream>
#include <vector>
#include <random>
#include <string>
#include <tuple>
#include <initializer_list>
#include <algorithm>

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
	void setData(size_t i, size_t j, float value) { _data[i][j]=value; };
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
			result = std::max(result,*max_element(_data[i].begin(), _data[i].end()));
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
				result._data[i][j] = this->_data[start+i][j];
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
				double exponemt = exp(matrix._data[i][j]-fMax);
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
	os << "])" << endl ;
	os << "  Shape " << matrix.shape() << endl;

	return os;
}

