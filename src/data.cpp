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
    vector<vector<double> > data; 
    ifstream input_file;
    input_file.open(path.c_str());
    while (input_file)
    {
        vector<string> string_file_data;
        string file_data;
        getline(input_file, file_data);
        try
        {
            boost::split(string_file_data, file_data, boost::is_any_of(","));
            vector<double> row;
            for (int i=1; i < string_file_data.size(); ++i) {
                row.push_back(boost::lexical_cast<double>(string_file_data[i]));
            }
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