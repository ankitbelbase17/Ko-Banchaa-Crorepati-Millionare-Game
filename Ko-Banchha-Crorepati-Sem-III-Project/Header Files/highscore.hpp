#include<iostream>
#include<SFML/Graphics.hpp>
#include "Filemanager.hpp"
class highscore {

	FileManager fileManager;
    sf::Sprite _sprite;
    sf::Text _text, _label, _input_text, _score;
    sf::Texture background;
    sf::Text __text, __name, __score;
    sf::Clock _clock;
    bool typing = true;
    bool showScore = true;
    std::string playerName;
    int score_value;
    sf::RectangleShape inputBox;
    std::vector<std::pair<std::string, int >> playerScore;

public:
	void display(sf::RenderWindow& window, int score);
	
};