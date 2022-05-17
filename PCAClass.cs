using System;
using Accord.Math.Decompositions;
using Accord.Math;
using Accord.IO;
using Accord.Statistics;

namespace PricipalComponentsAnalysis
{
    public class PCA
    {

        /*
        * Principal Component Analysis (PCA) algorithm 
        * Reduces data from n-dimensions to k-dimensions
        * 
        * Compute the eigen vectors of covariance matrix sigma where sigma = (X * X) (transpose)
        * The resulting X will be an n x n square matrix and 3 different matrices will be returned
        * by the SVD routine after processing X: [U,S,V], we want the first k vectors of the U matrix
        * which are the principal components. Call this U(reduce), then to generate the Z matrix
        * we compute U(reduce) transpose * X
        */

        // Public class properties
        public double[,] EigenVectors // The returned U Matrix aka principal components
        {
            get => _eigenVectors;
        }
        public int Rows
        {
            get => _inputMatrix.GetLength(0);
        }

        public int Columns
        {
            get => _inputMatrix.Columns();
        }
        public TimeSpan RunDuration
        {
            get => _runDuration;
        }

        public double[,] ZMatrix // The result matrix compressed to K dimensions
        {
            get => s_Z;
        }
        /* 
         * Private variables
         */

        private double[,] _eigenVectors;
        private TimeSpan _runDuration; // Returned duration of Singular Value decomposition
        private string _fileName;  // the name field for reading the input matrix provided as an argument to the contructor
        private double[,] _rawData;
        private static bool Intermediate = false;
        private double[,] _inputMatrix;
        private double[,] s_Z;
        


        // PCA class constructor
        public PCA(in string _Filename, in bool _writeIntermediate)
        {
            _fileName = _Filename;

            using (CsvReader reader = new CsvReader(_Filename, hasHeaders: false))
            {
                _rawData = reader.ToMatrix();
            }
            _inputMatrix = FeatureNormalization(_rawData);
            //Console.WriteLine("SVD run time {0}", SVDrun.Elapsed.ToString(@"mm\:ss"));
            Intermediate = _writeIntermediate;
            Console.WriteLine("foop");
            
        }

        public void Run()
        {
            // Stopwatch formatting
            // https://docs.microsoft.com/en-us/dotnet/api/system.diagnostics.stopwatch.elapsed?view=net-5.0

            System.Diagnostics.Stopwatch SVDrun = new System.Diagnostics.Stopwatch();
            SVDrun.Start();
            // Time how long the decomposition takes
            SingularValueDecomposition SVN = new SingularValueDecomposition(_inputMatrix/*_trythis._structRaw*/,
                               computeRightSingularVectors: true, computeLeftSingularVectors: false);
            _eigenVectors = SVN.RightSingularVectors;
            SVDrun.Stop();
            _runDuration = SVDrun.Elapsed;

            if (Intermediate) { WriteCSV(_fileName, _eigenVectors); }

        }

        public double[,] Compress(in int dimension)
        {
            // Using the set of previously calculated eigenvectors aka principal components, project the original Matrix
            // into a k x m dimensional matrix using the k subset of eigenvectors
            // Need to re-read the raw data as it is destroyed during the training operation.

            /*using (CsvReader reader = new CsvReader(_fileName, hasHeaders: false))
            {
                _rawData = reader.ToMatrix();
            }*/
            // What if we dont re read the input??
            _inputMatrix = FeatureNormalization(_rawData);
            s_Z = _inputMatrix.Dot(_eigenVectors.Get(startRow: 0, endRow: _eigenVectors.GetLength(0),
                                                         startColumn: 0, endColumn: dimension));
            
            return s_Z;
        }

        /**********************************************************************************************
        / Private routines
        /********************************************************************************************/

        private static double[,] FeatureNormalization(in double[,] localmatrix)
        {
            double[] Mu;
            double[] Sigma = Measures.StandardDeviation(localmatrix); // Calculate the standard deviation

            Mu = Mean(localmatrix);       // Calculate the column average

            for (int row = 0; row < localmatrix.GetLength(0); row++)
            {
                for (int col = 0; col < localmatrix.GetLength(1); col++)
                {
                    localmatrix[row, col] = localmatrix[row, col] - Mu[col];
                    localmatrix[row, col] = localmatrix[row, col] / Sigma[col];
                }
            }
            return (localmatrix);
        }

        private static double[] Mean(in double[,] input)
        {
            // Multidimensional array method

            const int ColumnSums = 0;

            double[] Mu = input.Mean(dimension: ColumnSums); // Calculates the column means
            return Mu;
        }

        private static void WriteCSV(in string FileName, in double[,] Matrix)
        {
            using (CsvWriter writer = new CsvWriter(System.Text.RegularExpressions.Regex.Replace(FileName, ".csv", "")
                  + "_EigenVectors.csv", delimiter: ','))
            {
                writer.Quote = ' '; // set the field separator to a space
                writer.Write(Matrix);
            }

        }


    }

}

