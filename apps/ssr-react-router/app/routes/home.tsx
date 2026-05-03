import { useState } from 'react';
import type { Route } from "./+types/home";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Hello World" },
    { name: "description", content: "Hello World page" },
  ];
}

export default function Home() {
  const [count, setCount] = useState(0);

  return (
    <main>
      <h1>Hello World</h1>
      <div>
        <p>Count: {count}</p>
        <button onClick={() => setCount(count + 1)}>Increment</button>
      </div>
    </main>
  );
}
