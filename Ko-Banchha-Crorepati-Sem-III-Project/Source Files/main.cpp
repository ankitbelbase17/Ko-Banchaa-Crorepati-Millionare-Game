#include <SFML/Audio.hpp>
#include "greetings.hpp"
#include "questions.hpp"
#include "players.hpp"
#include "rewards.hpp"
#include "texts.hpp"
#include "filemanager.hpp"
#include "congratulation.hpp"
using namespace std;
bool shuffleoptions = true;

enum class GameState
{
	Welcome,
	Greeting,
	ContentDisplay,
	GameOver,
	Congratulation
};

const sf::Vector2f WindowSize(900.f, 508.f);
sf::RenderWindow window(sf::VideoMode(WindowSize.x, WindowSize.y), "Ko Banchha Crorepati");

const unsigned int CharacterSize = 30;
const unsigned int GreetingCharacterSize = 50;
const unsigned int WelcomeCharacterSize = 50;

sf::Font font;
sf::Text text10("10     1 Crore", font, 25);
sf::Text text9(" 9      50,00,000", font, 25);
sf::Text text8(" 8      20,00,000", font, 25);
sf::Text text7(" 7      10,00,000", font, 25);
sf::Text text6(" 6      5,00,000", font, 25);
sf::Text text5(" 5      1,50,000", font, 25);
sf::Text text4(" 4      80,000", font, 25);
sf::Text text3(" 3      40,000", font, 25);
sf::Text text2(" 2      10,000", font, 25);
sf::Text text1(" 1      5,000", font, 25);
sf::Text textA("A", font, 35);
sf::Text textB("B", font, 35);
sf::Text textC("C", font, 35);
sf::Text textD("D", font, 35);

int main()
{
	GameState currentState = GameState::Welcome;
	if (!font.loadFromFile("Resources\\TT Bluescreens Trial Regular.ttf"))
	{
		std::cout << "Error loading font\n";
		return 1;
	}

	int reward = 0;

	welcomeText.setFont(font);
	welcomeText.setCharacterSize(50);
	welcomeText.setFillColor(sf::Color::White);
	welcomeText.setPosition(WelcomePosition);
	welcomeText.setString("WELCOME TO");

	congratulationText.setFont(font);
	congratulationText.setCharacterSize(50);
	congratulationText.setFillColor(sf::Color::White);
	congratulationText.setPosition(CongratulationPosition);

	gameoverText.setFont(font);
	gameoverText.setCharacterSize(50);
	gameoverText.setFillColor(sf::Color::White);
	gameoverText.setPosition(CongratulationPosition);

	greetingText.setFont(font);
	greetingText.setCharacterSize(GreetingCharacterSize);
	greetingText.setFillColor(sf::Color::White);
	greetingText.setPosition(GreetingPosition);
	greetingText.setString(greetings() + "! Let's Begin!");

	rewardText.setFont(font);
	rewardText.setCharacterSize(CharacterSize);
	rewardText.setFillColor(sf::Color::White);
	rewardText.setPosition(200, 235);

	quitText.setFont(font);
	quitText.setCharacterSize(35);
	quitText.setFillColor(sf::Color::White);
	quitText.setPosition(175, 50);
	quitText.setString("Are you sure you want to quit?");

	timerText.setFont(font);
	timerText.setCharacterSize(50);
	timerText.setFillColor(sf::Color::White);
	timerText.setPosition(430, 250);

	sf::Texture background;
	if (!background.loadFromFile("Resources\\Background.jpg"))
		return EXIT_FAILURE;
	sf::Sprite bg_sprite(background);

	sf::Texture logo;
	if (!logo.loadFromFile("Resources\\Logo.png"))
		return EXIT_FAILURE;
	sf::Sprite lg_sprite(logo);
	lg_sprite.setPosition(250, 50);

	sf::Texture qslogo;
	if (!qslogo.loadFromFile("Resources\\Logo.png"))
		return EXIT_FAILURE;
	sf::Sprite logo_sprite(qslogo);
	logo_sprite.setPosition(350, 10);
	logo_sprite.setScale(0.5f, 0.5f);

	sf::Music intro_music;
	if (!intro_music.openFromFile("Resources\\KBC Theme BGM.mp3"))
		return EXIT_FAILURE;
	sf::Music timer_music;
	if (!timer_music.openFromFile("Resources\\KBC Clock.mp3"))
		return EXIT_FAILURE;
	sf::Music applause_music;
	if (!applause_music.openFromFile("Resources\\KBC Applause.mp3"))
		return EXIT_FAILURE;
	sf::Music wrong_music;
	if (!wrong_music.openFromFile("Resources\\KBC Wrong.mp3"))
		return EXIT_FAILURE;

	std::vector<Button> welcome;
	welcome.emplace_back("Continue", ButtonPosition, font);

	std::vector<Button> next;
	next.emplace_back("Next", ButtonPosition, font);

	std::vector<Button> login;
	login.emplace_back("Login", ButtonPosition, font);

	std::vector<Button> quit;
	quit.emplace_back("Quit", quitPosition, font);

	std::vector<Button> exit;
	exit.emplace_back("Yes", yesPosition, font);
	exit.emplace_back("No", noPosition, font);

	bool focusUsername = true;


	intro_music.setLoop(true);
	intro_music.play();

	vector<Player> leaderboard;

	int currentQuestionIndex = 0;
	vector<Question> questions;
	vector<string> shuffledOptions;


	std::vector<sf::Text>questionTexts;

	text10.setPosition(700, 10);
	text9.setPosition(700, 35);
	text8.setPosition(700, 60);
	text7.setPosition(700, 85);
	text6.setPosition(700, 110);
	text5.setPosition(700, 135);
	text4.setPosition(700, 160);
	text3.setPosition(700, 185);
	text2.setPosition(700, 210);
	text1.setPosition(700, 235);

	textA.setPosition(105, 390);
	textB.setPosition(497, 390);
	textC.setPosition(105, 456);
	textD.setPosition(497, 456);

	text10.setFillColor(sf::Color::White);
	text9.setFillColor(sf::Color::White);
	text8.setFillColor(sf::Color::White);
	text7.setFillColor(sf::Color::White);
	text6.setFillColor(sf::Color::White);
	text5.setFillColor(sf::Color::White);
	text4.setFillColor(sf::Color::White);
	text3.setFillColor(sf::Color::White);
	text2.setFillColor(sf::Color::White);
	text1.setFillColor(sf::Color::White);

	textA.setFillColor(sf::Color(255, 215, 0));
	textB.setFillColor(sf::Color(255, 215, 0));
	textC.setFillColor(sf::Color(255, 215, 0));
	textD.setFillColor(sf::Color(255, 215, 0));


	//Highlight
	bool highlight = false;
	//Convex shapes
	sf::ConvexShape Qconvex;
	Qconvex.setPointCount(6);
	Qconvex.setPoint(0, sf::Vector2f(62.f, 337.f));
	Qconvex.setPoint(1, sf::Vector2f(108.f, 305.f));
	Qconvex.setPoint(2, sf::Vector2f(792.f, 305.f));
	Qconvex.setPoint(3, sf::Vector2f(840.f, 337.f));
	Qconvex.setPoint(4, sf::Vector2f(792.f, 370.f));
	Qconvex.setPoint(5, sf::Vector2f(108.f, 370.f));
	Qconvex.setFillColor(sf::Color(150, 50, 250));
	Qconvex.setOutlineThickness(-5.f);
	Qconvex.setOutlineColor(sf::Color(250, 150, 100));

	sf::ConvexShape Aconvex;
	Aconvex.setPointCount(6);
	Aconvex.setPoint(0, sf::Vector2f(62.f, 411.f));
	Aconvex.setPoint(1, sf::Vector2f(85.f, 382.f));
	Aconvex.setPoint(2, sf::Vector2f(425.f, 382.f));
	Aconvex.setPoint(3, sf::Vector2f(448.f, 411.f));
	Aconvex.setPoint(4, sf::Vector2f(425.f, 437.f));
	Aconvex.setPoint(5, sf::Vector2f(85.f, 437.f));
	Aconvex.setFillColor(sf::Color(150, 50, 250));
	Aconvex.setOutlineThickness(-5.f);
	Aconvex.setOutlineColor(sf::Color(250, 150, 100));

	sf::ConvexShape Bconvex;
	Bconvex.setPointCount(6);
	Bconvex.setPoint(0, sf::Vector2f(453.f, 411.f));
	Bconvex.setPoint(1, sf::Vector2f(477.f, 382.f));
	Bconvex.setPoint(2, sf::Vector2f(818.f, 382.f));
	Bconvex.setPoint(3, sf::Vector2f(842.f, 411.f));
	Bconvex.setPoint(4, sf::Vector2f(818.f, 437.f));
	Bconvex.setPoint(5, sf::Vector2f(477.f, 437.f));
	Bconvex.setFillColor(sf::Color(150, 50, 250));
	Bconvex.setOutlineThickness(-5.f);
	Bconvex.setOutlineColor(sf::Color(250, 150, 100));

	sf::ConvexShape Cconvex;
	Cconvex.setPointCount(6);
	Cconvex.setPoint(0, sf::Vector2f(63.f, 477.f));
	Cconvex.setPoint(1, sf::Vector2f(86.f, 448.f));
	Cconvex.setPoint(2, sf::Vector2f(426.f, 448.f));
	Cconvex.setPoint(3, sf::Vector2f(449.f, 477.f));
	Cconvex.setPoint(4, sf::Vector2f(426.f, 503.f));
	Cconvex.setPoint(5, sf::Vector2f(86.f, 503.f));
	Cconvex.setFillColor(sf::Color(150, 50, 250));
	Cconvex.setOutlineThickness(-5.f);
	Cconvex.setOutlineColor(sf::Color(250, 150, 100));

	sf::ConvexShape Dconvex;
	Dconvex.setPointCount(6);
	Dconvex.setPoint(0, sf::Vector2f(454.f, 477.f));
	Dconvex.setPoint(1, sf::Vector2f(478.f, 448.f));
	Dconvex.setPoint(2, sf::Vector2f(819.f, 448.f));
	Dconvex.setPoint(3, sf::Vector2f(843.f, 477.f));
	Dconvex.setPoint(4, sf::Vector2f(819.f, 503.f));
	Dconvex.setPoint(5, sf::Vector2f(478.f, 503.f));
	Dconvex.setFillColor(sf::Color(150, 50, 250));
	Dconvex.setOutlineThickness(-5.f);
	Dconvex.setOutlineColor(sf::Color(250, 150, 100));

	std::vector<optionButton> option;
	int currentLevel = 0;
	bool showConfirmation = false;

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			/*if (event.type == sf::Event::Closed)
				window.close();*/
			if (event.type == sf::Event::MouseButtonPressed)
			{
				if (event.mouseButton.button == sf::Mouse::Left)
				{
					sf::Vector2f mousePos(static_cast<float>(event.mouseButton.x), static_cast<float>(event.mouseButton.y));
					if (currentState == GameState::Welcome)
					{
						for (auto& button : welcome)
						{
							if (button.contains(mousePos))
							{
								if (currentState == GameState::Welcome)
								{
									button.isActive = true;
									if (button.text.getString() == "Continue")
									{
										currentState = GameState::Greeting;
									}
								}
							}
							else
							{
								button.isActive = false;
								button.shape.setFillColor(ButtonDefaultColor);
							}
						}
					}
					else if (currentState == GameState::Greeting)
					{
						for (auto& button : next)
						{
							if (button.contains(mousePos))
							{
								if (currentState == GameState::Greeting)
								{
									button.isActive = true;
									if (button.text.getString() == "Next")
									{
										intro_music.stop();
										timer_music.setLoop(true);
										timer_music.play();
										currentState = GameState::ContentDisplay;
									}
								}
							}
							else
							{
								button.isActive = false;
								button.shape.setFillColor(ButtonDefaultColor);
							}
						}
					}
					else if (currentState == GameState::ContentDisplay)
					{
						string levelFilename = "Resources\\level" + to_string(currentQuestionIndex) + ".txt";
						questions = loadQuestionsFromFile(levelFilename);
						window.clear();
						if (shuffleoptions) {
							//shuffle(questions.begin(), questions.end(), mt19937(random_device()()));
							if (currentQuestionIndex >= 0 && currentQuestionIndex < questions.size()) {

								Question& q = questions[currentQuestionIndex];
								questionText.setString(q.getQuestionText());
								shuffledOptions = q.getShuffledOptions();
								option.clear();
								option.emplace_back(shuffledOptions[0], optionAPosition, font);
								option.emplace_back(shuffledOptions[1], optionBPosition, font);
								option.emplace_back(shuffledOptions[2], optionCPosition, font);
								option.emplace_back(shuffledOptions[3], optionDPosition, font);
								for (auto& button : option) {
									button.shape.setFillColor(sf::Color::Transparent);
								}
								shuffleoptions = false;
							}
						}
						if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {

							sf::Vector2f mousePos(static_cast<float>(event.mouseButton.x), static_cast<float>(event.mouseButton.y));
							for (auto& button : option)
							{
								if (button.contains(mousePos) && !shuffleoptions) {
									if (questions[currentQuestionIndex].isCorrect(button.text.getString())) {
										button.shape.setFillColor(sf::Color::Green);
										reward = rewards[currentQuestionIndex + 1];
										if (reward == 10000000)
										{
											timer_music.stop();
											applause_music.play();
											CongratulationState congratulationState(font);
											congratulationState.display(window, reward);
										}
										rewardText.setString("Rewards earned - " + to_string(reward));
									}
									else {
										timer_music.stop();
										wrong_music.play();
										button.shape.setFillColor(sf::Color::Red);
										GameOverState gameoverState(font);
										gameoverState.display(window, reward);

									}
									currentQuestionIndex = (currentQuestionIndex + 1) % questions.size();
									shuffleoptions = true;
								}
							}
							for (auto& button : quit)
							{
								if (button.contains(mousePos)) {
									if (event.type == sf::Event::MouseButtonPressed && event.mouseButton.button == sf::Mouse::Left) {
										sf::RenderWindow confirmationWindow(sf::VideoMode(600, 300), "Confirmation");
										confirmationWindow.setPosition(sf::Vector2i(window.getPosition().x + 300, window.getPosition().y + 150));
										confirmationWindow.clear();
										while (confirmationWindow.isOpen())
										{
											sf::Event evnt;
											while (confirmationWindow.pollEvent(evnt))
											{
												if (evnt.type == sf::Event::Closed)
													confirmationWindow.close();
												if (evnt.type == sf::Event::MouseButtonPressed)
												{
													if (evnt.mouseButton.button == sf::Mouse::Left)
													{
														sf::Vector2f mousePos(static_cast<float>(evnt.mouseButton.x), static_cast<float>(evnt.mouseButton.y));
														for (auto& tbutton : exit)
															if (tbutton.contains(mousePos))
															{
																if (tbutton.text.getString() == "Yes")
																{
																	confirmationWindow.close();
																	window.close();
																}
																else if (tbutton.text.getString() == "No")
																{
																	confirmationWindow.close();
																}

															}
													}
													showConfirmation = true;
												}
											}
											confirmationWindow.draw(bg_sprite);
											confirmationWindow.draw(quitText);
											for (auto& tbutton : exit)
											{
												confirmationWindow.draw(tbutton.shape);
												confirmationWindow.draw(tbutton.text);
											}

											confirmationWindow.display();
										}

									}
								}

								for (auto& button : quit)
								{
									window.draw(button.shape);
									window.draw(button.text);

								}

							}
						}
					}
				}
			}
		}
		window.clear();
		if (currentState == GameState::Welcome)
		{
			window.draw(bg_sprite);
			window.draw(lg_sprite);
			window.draw(welcomeText);
			for (const auto& button : welcome)
			{
				window.draw(button.shape);
				window.draw(button.text);
			}
		}
		else if (currentState == GameState::Greeting)
		{
			window.draw(bg_sprite);
			window.draw(lg_sprite);
			window.draw(greetingText);
			for (const auto& button : next)
			{
				window.draw(button.shape);
				window.draw(button.text);
			}
		}
		else if (currentState == GameState::ContentDisplay)
		{
			window.clear();
			questionText.setFont(font);
			questionText.setCharacterSize(CharacterSize);
			questionText.setFillColor(sf::Color::White);
			questionText.setPosition(QuestionPosition);
			window.clear();
			window.draw(resultText);
			window.draw(bg_sprite);
			window.draw(logo_sprite);
			window.draw(text10);
			window.draw(text9);
			window.draw(text8);
			window.draw(text7);
			window.draw(text6);
			window.draw(text5);
			window.draw(text4);
			window.draw(text3);
			window.draw(text2);
			window.draw(text1);
			window.draw(rewardText);

			window.draw(Qconvex);
			window.draw(Aconvex);
			window.draw(Bconvex);
			window.draw(Cconvex);
			window.draw(Dconvex);

			window.draw(textA);
			window.draw(textB);
			window.draw(textC);
			window.draw(textD);

			window.draw(questionText);
			for (const auto& button : option)
			{
				window.draw(button.shape);
				window.draw(button.text);
			}
			for (const auto& button : quit)
			{
				window.draw(button.shape);
				window.draw(button.text);
			}
		}
		window.display();
	}

}