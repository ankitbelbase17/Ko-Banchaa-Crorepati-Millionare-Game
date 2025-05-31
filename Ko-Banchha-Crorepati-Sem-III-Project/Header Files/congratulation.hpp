#ifndef CONGRATULATION_HPP
#define CONGRATULATION_HPP

#include <SFML/Graphics.hpp>
#include <iostream>
#include <string>
#include "gameover.hpp"
class CongratulationState {
public:
    CongratulationState(sf::Font& font);
    void display(sf::RenderWindow& window, int reward);

private:
    sf::Text congratulationText;
};

#endif // CONGRATULATION_HPP
