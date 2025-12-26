# Retail-GPT 🛍️

An agent that creates a dynamic, personalized shopping experience from end to end. Retail-GPT generates unique product images, crafts custom descriptions, and designs a virtual storefront that is entirely tailored to a user's individual style, preferences, and aesthetic.

-----

## ✨ Features

  * **Hyper-Personalized Shopping Experience:** Goes beyond simple recommendations to create a fully customized and unique shopping environment for each user.
  * **AI-Powered Product Imaging:** Utilizes a advanced **diffusion model** to generate hyper-realistic product images on demand, allowing users to visualize products in custom scenarios (e.g., a dress on a beach, a shirt in a city).
  * **Dynamic Content Generation:** A **Copywriting Agent** automatically writes personalized product descriptions that resonate with the user's unique style and interests.
  * **Intelligent Visual Merchandising:** A **Visual Merchandiser Agent** creates a virtual storefront layout tailored to the user's historical preferences and style analysis.
  * **Style Analysis:** A **Stylist Agent** analyzes a user's style from their social media profiles to understand their fashion preferences.
  * **Aesthetic Vector Space:** Uses **Qdrant** to create a high-dimensional vector space of styles, brands, and aesthetics to power highly accurate and personalized recommendations.

-----

## ⚙️ Tech Stack

  * **Frontend:** [Next.js](https://nextjs.org/)
  * **Text Generation:** [Gemma](https://huggingface.co/google/gemma-7b)
  * **Image Generation:** Large-scale generative model (e.g., Imagen 4)
  * **Vector Search:** [Weaviate](https://qdrant.tech/)

-----

## 🚀 Getting Started

### Prerequisites

  * Node.js (for Next.js)
  * Docker (for Qdrant)
  * Access to your chosen text and image generation models

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/saadsalmanakram/Retail-GPT.git
    cd Retail-GPT
    ```
2.  **Set up the front-end:**
    ```bash
    cd frontend
    npm install
    ```
3.  **Start Docker containers:**
    Use the provided `docker-compose.yml` to run the Qdrant instance.

### Configuration

Create a `.env` file in the `frontend` directory and add your API keys for the generative models.

### Usage

Run the Next.js server and connect the front-end to your back-end agents to begin generating personalized content.

```bash
npm run dev
```

-----

 