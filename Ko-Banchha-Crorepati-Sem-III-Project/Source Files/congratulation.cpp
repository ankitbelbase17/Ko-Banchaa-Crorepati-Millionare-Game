#include "congratulation.hpp"

CongratulationState::CongratulationState(sf::Font& font) {
    congratulationText.setFont(font);
    congratulationText.setCharacterSize(50);
    congratulationText.setFillColor(sf::Color::White);
    congratulationText.setPosition(CongratulationPosition);
    congratulationText.setString("Congratulations!!! You are now a CrorePati!");
}

void CongratulationState::display(sf::RenderWindow& window, int reward) {
    sf::Texture background;
    if (!background.loadFromFile("Resources\\Background.jpg"))
    {
        std::cout << "Error loading background\n";
    }
    sf::Sprite bg_sprite(background);

    sf::Font font;
    if (!font.loadFromFile("Resources\\TT Bluescreens Trial Regular.ttf"))
    {
        std::cout << "Error loading font\n";
    }

	std::vector<Button> next;
	next.emplace_back("Next", ButtonPosition, font);

    // Clear the window
    window.clear();
    window.draw(bg_sprite);

	while (window.isOpen())
	{
		// Clear the window
		window.clear();
		window.draw(bg_sprite);

		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();

			if (event.mouseButton.button == sf::Mouse::Left)
			{
				sf::Vector2f mousePos(static_cast<float>(event.mouseButton.x), static_cast<float>(event.mouseButton.y));
				for (auto& button : next)
				{
					if (button.contains(mousePos))
					{
						if (button.text.getString() == "Next")
						{
							highscore h;
							h.display(window, reward);
						}
					}
				}
			}
		}
		window.draw(congratulationText);
		for (auto& button : next)
		{
			window.draw(button.shape);
			window.draw(button.text);
		}
		window.display();
	}

}
