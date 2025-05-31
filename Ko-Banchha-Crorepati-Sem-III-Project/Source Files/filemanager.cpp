#include"filemanager.hpp"
FileManager::FileManager()
{
    std::ifstream file("Resources\\score.txt");
    if (file.is_open()) {}
    else
    {
        std::cout << "Unable to open file1";
    }
}

FileManager::~FileManager()
{
}

void FileManager::storeScore(std::string playerName, int score)
{
    std::ofstream file;
    file.open("Resources\\score.txt", std::ios::app);
    if (file.fail())
    {
        std::cout << "Error opening file" << std::endl;
    }
    file << playerName << " " << score << std::endl;
    file.close();
}

// Store top nine scores in map
void FileManager::getScoreFromFile()
{
    std::ifstream file("Resources\\score.txt");
    if (file.is_open())
    {
        std::string line;
        while (getline(file, line))
        {
            std::string name = line.substr(0, line.find(' '));
            std::string score = line.substr(name.length() + 1, line.length());
            playerScore[name] = std::stoi(score); // Store name and score in map for sorting
        }
        file.close();

    }
    else
    {
        std::cout << "Unable to open file";
    }
}

std::map <std::string, int> FileManager::getPlayerScore() {
    // Get an iterator pointing to the first element in the map
    std::map<std::string, int>::iterator it = playerScore.begin();

    // Iterate through the map and print the elements
    while (it != playerScore.end())
    {
        ++it;
    }
    return playerScore;
}

std::vector<std::pair<std::string, int>> FileManager::GetSortedScore()
{
    std::vector<std::pair<std::string, int>> SortedPlayerScore(playerScore.begin(), playerScore.end());
    std::sort(SortedPlayerScore.begin(), SortedPlayerScore.end(), FileManager::CompareScore);
    return SortedPlayerScore;
}

bool FileManager::CompareScore(const std::pair<std::string, int>& playerscore1,
    const std::pair<std::string, int>& playerscore2) {
    return playerscore1.second > playerscore2.second;
}
