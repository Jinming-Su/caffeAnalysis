#include <iostream>
using namespace std;

#define RegisterBrewFunction() \
	(cout << "123" << endl;)\
}


RegisterBrewFunction();


int main(int argc, char* argv[]) {

  cout << argc << endl;
  cout << argv[0] << endl;
}
