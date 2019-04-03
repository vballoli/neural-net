#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <boost/algorithm/string.hpp>
#include <boost/lexical_cast.hpp>
#include "data.hpp"

using namespace std;

Data::Data()
{
    cout<<"Initialized data"<<endl;
}

Data::~Data()
{
}

vector<vector<double>> Data::formatData(string path)
{
    vector<vector<double> > data{ { 1, 2 }, 
                               { 4, 5, 6 }, 
                               { 7, 8, 9, 10 } }; 
    ifstream input_file;
    data.clear();
    input_file.open(path.c_str());
    while (input_file)
    {
        vector<string> string_file_data;
        string file_data;
        getline(input_file, file_data);
        boost::split(string_file_data, file_data, boost::is_any_of(","));
        try
        {
            double x1 = boost::lexical_cast<double>(string_file_data[1]);
            double x2 = boost::lexical_cast<double>(string_file_data[2]);
            double y = boost::lexical_cast<double>(string_file_data[3]);
            vector<double> row(3,3);
            row.clear();
            row.push_back(x1);
            row.push_back(x2);
            row.push_back(y);
            data.push_back(row);
            cout<<data.back()[0]<<endl;
        } catch (exception e) {
            continue;
        }
    }
    cout<<"Size"<<"\t";
    cout<<data.size()<<endl;
    input_file.close();
    return data;
}