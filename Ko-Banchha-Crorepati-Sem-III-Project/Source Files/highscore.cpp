#include"highscore.hpp"
void highscore::display(sf::RenderWindow& window, int score)
{
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

	sf::Text hs;
	hs.setString("High Score");
	hs.setCharacterSize(36);
	hs.setFont(font);
	hs.setPosition(250.f, 50.f);

	_score.setFont(font);
	_score.setString("Score: " + std::to_string(score));
	_score.setCharacterSize(30);
	_score.setFillColor(sf::Color::White);
	_score.setPosition(561, 200);

	// Ask User information

	_label.setFont(font);
	_label.setString("Name:");
	_label.setCharacterSize(36);
	_label.setPosition(100, 150);

	inputBox.setPosition(100, 200);
	inputBox.setFillColor(sf::Color::White);
	inputBox.setOutlineThickness(2);
	inputBox.setOutlineColor(sf::Color::Black);
	inputBox.setSize(sf::Vector2f(300, 40));

	_input_text.setFont(font);
	_input_text.setCharacterSize(24);
	_input_text.setFillColor(sf::Color::Black);
	_input_text.setPosition(100, 200);

	while (window.isOpen())
	{
		window.clear();
		window.draw(bg_sprite);
		sf::Event evt;
		while (window.pollEvent(evt))
		{
			if (evt.type == sf::Event::Closed)
				window.close();
			if (evt.type == sf::Event::TextEntered && typing)
			{
				if (evt.text.unicode == 32)
				{ // Enter key
					typing = false;
					showScore = false;
					// Clear the input field
				}
				if (evt.text.unicode == 8 && !playerName.empty())
				{ // Backspace
					playerName.pop_back();
				}
				if (evt.text.unicode >= 32 && evt.text.unicode <= 126)
				{
					playerName += static_cast<char>(evt.text.unicode);
				}
				_input_text.setString(playerName);
			}
		}
		int i = 0;

		while (!typing)
		{
			fileManager.storeScore(playerName, score);

			fileManager.getScoreFromFile();
			playerScore = fileManager.GetSortedScore();
			for (const auto& player : playerScore)
			{
				__name.setString(std::to_string(i + 1) + ". " + player.first);
				__name.setFont(font);
				__name.setFillColor(sf::Color::White);
				__name.setCharacterSize(26);
				__name.setPosition(50, 50 + i * 50);

				__score.setString(std::to_string(player.second));
				__score.setFont(font);
				__score.setCharacterSize(26);
				__score.setPosition(200, 50 + i * 50);
				__score.setString(std::to_string(player.second));

				window.draw(__name);
				window.draw(__score);
				if (i == 9)
					break;
				i++;
			}
			if (i == 9)
				break;
		}
		if (showScore)
		{
			window.draw(__name);
			window.draw(__score);
			window.draw(_label);
			window.draw(inputBox);
			window.draw(_input_text);
			window.draw(_score);
			window.draw(hs);
		}
		window.display();
	}
}