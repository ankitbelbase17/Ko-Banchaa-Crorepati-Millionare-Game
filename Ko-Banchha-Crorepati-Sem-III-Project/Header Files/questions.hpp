#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <random>
using namespace std;
class Question
{
private:
	string questionText;
	vector<string> options;
	string correctOption;
public:
	Question();
	Question(string questionText, vector<string> options, string correctOption);
	string getQuestionText() const;
	vector<string> getShuffledOptions() const;
	bool isCorrect(string choice) const;
};
vector <Question> loadQuestionsFromFile(const string& filename);