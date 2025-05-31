#include<SFML/Graphics.hpp>

const sf::Vector2f ButtonSize(100.f, 50.f);
const sf::Vector2f welcomeButtonSize(150.f, 50.f);
const sf::Color ButtonDefaultColor(255, 0, 255);
const sf::Vector2f ButtonPosition(700.f, 380.f);
const sf::Vector2f GameOverPosition(350.f, 50.f);
const sf::Vector2f quitPosition(750.f, 262.f);
const sf::Vector2f WelcomePosition(390.f, 0.f);
const sf::Vector2f CongratulationPosition(240.f, 100.f);
const sf::Vector2f QuestionPosition(130.f, 310.f);
const sf::Vector2f optionAPosition(145.f, 385.f);
const sf::Vector2f optionBPosition(537.f, 385.f);
const sf::Vector2f optionCPosition(145.f, 451.f);
const sf::Vector2f optionDPosition(537.f, 451.f);
const sf::Vector2f GreetingPosition(30.f, 30.f);
const sf::Vector2f yesPosition(126.f, 200.f);
const sf::Vector2f noPosition(375.f, 200.f);

class Button
{
public:
	sf::RectangleShape shape;
	sf::Text text;
	bool isActive;
	Button(const std::string& buttonText, const sf::Vector2f& buttonPosition, sf::Font& font);
	bool contains(const sf::Vector2f& mousePos);
};

class optionButton
{
public:
	sf::RectangleShape shape;
	sf::Text text;
	bool isActive;
	optionButton(const std::string& buttonText, const sf::Vector2f& buttonPosition, sf::Font& font);
	bool contains(const sf::Vector2f& mousePos);
};
