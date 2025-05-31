#include "questions.hpp"
Question::Question() {}
Question::Question(string questionText, vector<string> options, string correctOption)
{
	this->questionText = questionText;
	this->options = options;
	this->correctOption = correctOption;
}
string Question::getQuestionText() const
{
	return questionText;
}
vector<string> Question::getShuffledOptions() const
{
	vector<string> shuffledOptions = options;
	shuffle(shuffledOptions.begin(), shuffledOptions.end(), mt19937(random_device()()));
	return shuffledOptions;
}
bool Question::isCorrect(string choice) const
{
	return correctOption == choice;
}
vector <Question> loadQuestionsFromFile(const string& filename)
{
	vector<Question> questions;
	ifstream fin(filename);
	if (!fin)
	{
		cout << "Could not open file: " << filename << endl;
		return questions;
	}
	string line;
	while (getline(fin, line))
	{
		size_t questionEnd = line.find("?");
		if (questionEnd == string::npos)
		{
			continue;
		}
		string questionText = line.substr(0, questionEnd + 1);
		line = line.substr(questionEnd + 1);
		vector<string> options;
		for (size_t i = 0; i < 4; ++i)
		{
			size_t optionStart = line.find('(');
			size_t optionEnd = line.find(')');
			if (optionStart == string::npos || optionEnd == string::npos)
			{
				break;
			}
			string option = line.substr(optionStart + 1, optionEnd - optionStart - 1);
			options.push_back(option);
			line = line.substr(optionEnd + 1);
		}
		if (line.empty() || line[0] != '.')
		{
			continue;
		}
		string correctOption = line.substr(1);
		questions.push_back(Question(questionText, options, correctOption));

	}
	fin.close();
	return questions;
}