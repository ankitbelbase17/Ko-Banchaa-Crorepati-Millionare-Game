#include<vector>
#include<algorithm>
#include<random>
using namespace std;
class Player
{
public:
	string name;
	int score;
	Player(const string& name, int score) : name(name), score(score) {}
};
bool comparePlayers(const Player& player1, const Player& player2)
{
	return player1.score > player2.score;
}