#ifndef GAMEOVER_HPP
#define GAMEOVER_HPP

#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include "buttons.hpp"
#include "HighScore.hpp"

class GameOverState {
public:
	GameOverState(sf::Font& font);
	void display(sf::RenderWindow& window, int reward);
private:
	sf::Text gameoverText;
};

#endif // GAMEOVER_HPP