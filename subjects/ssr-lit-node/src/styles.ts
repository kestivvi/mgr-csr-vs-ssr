import { css } from 'lit';

export const globalStyles = css`
  h1 {
      font-size: 2.5rem;
      font-weight: 700;
      margin-bottom: 1rem;
      color: #222;
  }

  p {
      font-size: 1.2rem;
      color: #666;
      margin-bottom: 1.5rem;
  }

  button {
      background: #007bff;
      color: white;
      border: none;
      padding: 0.8rem 1.5rem;
      font-size: 1rem;
      font-weight: 600;
      border-radius: 6px;
      cursor: pointer;
      transition: background 0.2s, transform 0.1s;
  }

  button:hover {
      background: #0056b3;
  }

  button:active {
      transform: scale(0.98);
  }
`;
