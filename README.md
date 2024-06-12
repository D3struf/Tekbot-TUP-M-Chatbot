# Tekbot-TUP-M-Chatbot

Tekbot is a virtual assistant developed for TUP-Manila students to assist with academic management through natural language processing (NLP) techniques.

## Project Overview

This project aims to address the academic management challenges faced by TUP-Manila students by creating a chatbot that provides information on various academic and administrative topics.

## Table of Contents

- [Tekbot-TUP-M-Chatbot](#tekbot-tup-m-chatbot)
  - [Project Overview](#project-overview)
  - [Table of Contents](#table-of-contents)
  - [Technologies Used](#technologies-used)
  - [System Architecture](#system-architecture)
  - [Usage](#usage)
  - [Installation](#installation)
  - [Contributing](#contributing)
  - [License](#license)
  - [Authors](#authors)

## Technologies Used

- Flask
- CSS
- JavaScript
- PythonAnywhere

## System Architecture

- **User Interaction:** Interface for students, administrative staff, and teachers.
- **Chatbot Server:**
  - Intent and Entity Classification
  - Response Generation
  - Natural Language Processing
- **NLP Techniques:** Classifies user queries.
- **Knowledge Base:** Stores templates and data for responses.
- **Feedback Loop:** Collects and utilizes user feedback.
- **Integration with Existing Systems:** Uses APIs for communication and action execution.

## Usage

- Access the chatbot via the web interface. [Tekbot](tekbot.pythonanywhere.com)
- Interact with the bot by typing queries related to TUP-Manila academic and administrative information.

## Installation

- Clone the repository:

``` bash
git clone https://github.com/D3struf/Tekbot-TUP-M-Chatbot.git
```

- Navigate to the project directory:

``` bash
cd Tekbot-TUP-M-Chatbot
```

- Install Dependencies:

``` bash
pip install -r requirements.txt
pip install spacy
python -m spacy download en_core_web_sm
```

- Run the application:

``` bash
flask run
```

## Contributing

Contributions are welcome! Please fork the repository and create a pull request with your changes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Authors

[John Paul Monter](https://github.com/D3struf) - Lead Developer \
[Maria Evita Juan](https://github.com/evyjuan) - UI/UX  \
[Jeanne May Carolino](https://github.com/jeannmaycarolino) - Data Analyst & Research Developer \
[Mary Jane Calulang](https://github.com/meri-hane) - Project Manager \
[Vincent Johanne Tenorio](https://github.com/Yuhan-BSCS) - Team Lead
