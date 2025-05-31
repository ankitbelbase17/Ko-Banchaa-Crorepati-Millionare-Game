#pragma once
#include<SFML/Graphics.hpp>
#include<iostream>
#include"buttons.hpp"
std::vector<welcomeButton> welcome;
sf::Texture background;
sf::Sprite bg_sprite(background);
sf::Texture logo;
sf::Sprite lg_sprite(logo);
sf::Text welcomeText;
sf::Font font;
const unsigned int WelcomeCharacterSize = 50;
class Welcome {
public:
	Welcome() {};
	~Welcome() {};
	void display(sf::RenderWindow& window);
};