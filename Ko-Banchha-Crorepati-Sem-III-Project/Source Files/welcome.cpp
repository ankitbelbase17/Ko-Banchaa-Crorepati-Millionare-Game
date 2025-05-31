//#include"welcome.hpp"
//
//void Welcome::display(sf::RenderWindow& window)
//{
//	if (!background.loadFromFile("D:\\KBC Resources\\Images\\Background.jpg"))
//		std::cout << "Background could not be loaded\n";
//	if (!logo.loadFromFile("D:\\KBC Resources\\Images\\Logo.png"))
//		std::cout << "Logo could not be loaded\n";
//	if (!font.loadFromFile("D:\\KBC Resources\\Fonts\\TT Bluescreens Trial Regular.ttf"))
//		std::cout << "Error loading font\n";
//	lg_sprite.setPosition(250, 50);
//	welcomeText.setFont(font);
//	welcomeText.setCharacterSize(WelcomeCharacterSize);
//	welcomeText.setFillColor(sf::Color::White);
//	welcomeText.setPosition(WelcomePosition);
//	welcomeText.setString("WELCOME TO KBC!");
//
//	while (window.isOpen()) {
//		window.clear();
//		window.draw(bg_sprite);
//		sf::Event event;
//		while (window.pollEvent(event))
//		{
//			if (event.type == sf::Event::Closed)
//				window.close();
//			if (event.type == sf::Event::MouseButtonPressed)
//			{
//				if (event.mouseButton.button == sf::Mouse::Left)
//				{
//					sf::Vector2f mousePos(static_cast<float>(event.mouseButton.x), static_cast<float>(event.mouseButton.y));
//					for (auto& button : welcome)
//					{
//						if (button.contains(mousePos))
//						{
//								button.isActive = true;
//								if (button.text.getString() == "Continue")
//								{
//									currentState = GameState::Login;
//								}
//						}
//						else
//						{
//							button.isActive = false;
//							button.shape.setFillColor(ButtonDefaultColor);
//						}
//					}
//				}
//			}
//		}
//	}
//
//}