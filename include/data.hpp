#include <vector>
#include <string>

using namespace std;

class Data
{
private:
    
public:
    Data();
    ~Data();

    /*! 
    *   Formats the file into 2-dimensional vector with feature data and labels
    * 
    *   \param path - Path to input file(csv)
    */
    vector<vector<double>> formatData(string path);

protected:
    string path;

};