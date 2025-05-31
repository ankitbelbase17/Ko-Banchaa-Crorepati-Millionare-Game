#include "buttons.hpp"
Button::Button(const std::string& buttonText, const sf::Vector2f& buttonPosition, sf::Font& font)
	: isActive(false)
{
	shape.setSize(welcomeButtonSize);
	shape.setPosition(buttonPosition);
	shape.setFillColor(ButtonDefaultColor);

	text.setString(buttonText);
	text.setFont(font);
	text.setCharacterSize(24);
	text.setFillColor(sf::Color::Black);
	text.setPosition(
		buttonPosition.x + (welcomeButtonSize.x - text.getLocalBounds().width) / 2.f,
		buttonPosition.y + (welcomeButtonSize.y - text.getLocalBounds().height) / 2.f);
}

bool Button::contains(const sf::Vector2f& mousePos)
{
	return shape.getGlobalBounds().contains(mousePos);
}

optionButton::optionButton(const std::string& buttonText, const sf::Vector2f& buttonPosition, sf::Font& font)
	: isActive(false)
{
	shape.setSize(welcomeButtonSize);
	shape.setPosition(buttonPosition);
	shape.setFillColor(sf::Color::Transparent);

	text.setString(buttonText);
	text.setFont(font);
	text.setCharacterSize(26);
	text.setFillColor(sf::Color::White);
	text.setPosition(
		buttonPosition.x + (welcomeButtonSize.x - text.getLocalBounds().width) / 2.f,
		buttonPosition.y + (welcomeButtonSize.y - text.getLocalBounds().height) / 2.f);
}

bool optionButton::contains(const sf::Vector2f& mousePos)
{
	return shape.getGlobalBounds().contains(mousePos);
}