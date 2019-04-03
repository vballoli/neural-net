#include <vector>
#include <string>

using namespace std;

class Data
{
private:
    
public:
    Data();
    ~Data();
    vector<vector<double>> formatData(string path);

protected:
    string path;

};