#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <iostream>

namespace py = pybind11;

using std::ostream;
using std::vector;
using std::cout;
using std::endl;

template<typename T>
class Matrix{
    public:
		Matrix(T* _data,size_t _n,size_t _m,bool _external):
            data(_data),n(_n),m(_m),external(_external){
		}
        Matrix(const T* _data,size_t _n,size_t _m):
            n(_n),m(_m){
				data = new T[n*m];
				memcpy(data,_data,n*m*sizeof(T));
			}
        Matrix(size_t _n,size_t _m):n(_n),m(_m){
            data = new T[n*m];
        }
		Matrix(Matrix&& other) noexcept{
			*this = std::move(other);
		}
		Matrix& operator=(Matrix&& other)noexcept{
			if (this != &other){
				this->data = other.data;
				this->n = other.n;
				this->m = other.m;
				this->external = other.external;
				other.data = nullptr;
			}
			return *this;
		}
		~Matrix(){
			if (!external && data){
				delete[] data;
			}
		}
        template<typename U>
        Matrix dot(const Matrix<U> &matrix){
            if (m != matrix.n){
                throw std::runtime_error("Matrix dimensions do not match");
            }
            Matrix c(n,matrix.m);
            for(size_t i=0;i<n;i++){
                for(size_t j=0;j<matrix.m;j++){
                    T sum = 0;
                    for(size_t k=0;k<m;k++){
                        sum += (*this)(i,k)*matrix(k,j);
                    }
                    c(i,j) = sum;
                }
            }
            return c;
        }
		inline T operator()(size_t i,size_t j) const{
				return data[i*m+j];
		}
		inline T& operator()(size_t i,size_t j){
				return data[i*m+j];
		}
		// void apply_inplace(T (*f)(T) )const{
		// 	for(size_t i=0;i<n;i++){
		// 		for(size_t j=0;j<m;j++){
		// 			data[i*m+j] = f((*this)(i,j));
		// 		}
		// 	}
		// }
        template<typename U>
        Matrix& operator+=(const Matrix<U> &matrix){
            /**
             * @brief inplace addition
             * 
             */
            if (m != matrix.m || n != matrix.n){
                throw std::runtime_error("Matrix dimensions do not match");
            }
            for(size_t i=0;i<n;i++){
                for(size_t j=0;j<m;j++){
                    this->data[i*m+j] += matrix(i,j);
                }
            }
            return *this;
        }
		template<typename U>
        Matrix& operator-=(const Matrix<U> &matrix){
            /**
             * @brief inplace addition
             * 
             */
            if (m != matrix.m || n != matrix.n){
                throw std::runtime_error("Matrix dimensions do not match");
            }
            for(size_t i=0;i<n;i++){
                for(size_t j=0;j<m;j++){
                    this->data[i*m+j] -= matrix(i,j);
                }
            }
            return *this;
        }
		template<typename U>
        Matrix& operator/=(const U value){
			for(size_t i=0;i<n;i++){
                for(size_t j=0;j<m;j++){
                    this->data[i*m+j] /= value;
                }
            }
            return *this;
        }
		template<typename U>
        Matrix& operator*=(const U value){
            /**
             * @brief inplace addition
             * 
             */
            for(size_t i=0;i<n;i++){
                for(size_t j=0;j<m;j++){
                    this->data[i*m+j] *= value;
                }
            }
            return *this;
        }
		void softmax(){
			/**
			 * @brief apply softmax inplace
			 * 
			 */
			vector<T> sums(n);
			for(size_t i=0;i<n;i++){
				T sum = 0;
				for(size_t j=0;j<m;j++){
					data[i*m+j] = exp(data[i*m+j]);
					sum += data[i*m+j];
				}
				sums[i] = sum;
			}
			for(size_t i=0;i<n;i++){
				for(size_t j=0;j<m;j++){
					data[i*m+j] /= sums[i];
				}
			}
		}
		Matrix& tran(){
			/**
			 * @brief transpose inplace
			 * new shape is (m,n)
			 */
			T tmp[n*m];
			memcpy(tmp,data,n*m*sizeof(T));
			for(size_t i=0;i<m;i++){
				for(size_t j=0;j<n;j++){
					data[i*n+j] = tmp[j*m+i];
				}
			}
			std::swap(n,m);
			return *this;
		}
        T* data;
        size_t n;
        size_t m;
		bool external = false;
};
Matrix<unsigned char> construct_onehot_matrix(const unsigned char y[],size_t n,size_t m){
	/**
	 * @brief construct an onehot matrix from a vector of labels
	 * 
	 * @arg y: a vector of labels
	 * @arg n: the number of labels
	 * @arg m: the number of classes
	 * 
	 * @return Matrix(number_of_labels,number_of_classes)
	 */
	Matrix<unsigned char> onehot(n,m);
	memset(onehot.data,0,sizeof(unsigned char)*n*m);
	for (size_t i=0;i<n;i++){
		onehot(i,y[i]) = 1;
	}
	return onehot;
}
// template<typename T>
// ostream& operator<<(ostream& os, const Matrix<T>& matrix){
// 	for(size_t i=0;i<matrix.n;i++){
// 		for(size_t j=0;j<matrix.m;j++){
// 			os<<matrix(i,j)<<" ";
// 		}
// 		os<<endl;
// 	}
// 	return os;
// }
void softmax_regression_epoch_cpp(const float *X, const unsigned char *y,
								  float *theta_, size_t m, size_t n, size_t k,
								  float lr, size_t batch)
{
    /**
     * A C++ version of the softmax regression epoch code.  This should run a
     * single epoch over the data defined by X and y (and sizes m,n,k), and
     * modify theta in place.  Your function will probably want to allocate
     * (and then delete) some helper arrays to store the logits and gradients.
     *
     * Args:
     *     X (const float *): pointer to X data, of size m*n, stored in row
     *          major (C) format
     *     y (const unsigned char *): pointer to y data, of size m
     *     theta (float *): pointer to theta data, of size n*k, stored in row
     *          major (C) format
     *     m (size_t): number of examples
     *     n (size_t): input dimension
     *     k (size_t): number of classes
     *     lr (float): learning rate / SGD step size
     *     batch (int): SGD minibatch size
     *
     * Returns:
     *     (None)
     */

    /// BEGIN YOUR CODE
    Matrix<float> theta(theta_,n,k,true);
    for(size_t i =0;i<m;i+=batch){
        auto this_batch = std::min(batch,m-i);
        Matrix<float> x_batch(X+i*n,this_batch,n);
        auto Z = x_batch.dot(theta);
        Z.softmax();
        auto Iy = construct_onehot_matrix(y+i,this_batch,k);
        Z -= Iy;
        auto diff_softmax = x_batch.tran().dot(Z);
        diff_softmax /= this_batch;
        diff_softmax *= lr;
        theta -= diff_softmax;
    }
    /// END YOUR CODE
}


/**
 * This is the pybind11 code that wraps the function above.  It's only role is
 * wrap the function above in a Python module, and you do not need to make any
 * edits to the code
 */
PYBIND11_MODULE(simple_ml_ext, m) {
    m.def("softmax_regression_epoch_cpp",
    	[](py::array_t<float, py::array::c_style> X,
           py::array_t<unsigned char, py::array::c_style> y,
           py::array_t<float, py::array::c_style> theta,
           float lr,
           int batch) {
        softmax_regression_epoch_cpp(
        	static_cast<const float*>(X.request().ptr),
            static_cast<const unsigned char*>(y.request().ptr),
            static_cast<float*>(theta.request().ptr),
            X.request().shape[0],
            X.request().shape[1],
            theta.request().shape[1],
            lr,
            batch
           );
    },
    py::arg("X"), py::arg("y"), py::arg("theta"),
    py::arg("lr"), py::arg("batch"));
}
