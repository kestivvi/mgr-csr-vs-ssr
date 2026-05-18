import { useState } from 'react';
import type { Route } from "./+types/route";

export function meta({ params }: Route.MetaArgs) {
  return [
    { title: "Dynamic Page" },
    { name: "description", content: "Dynamic page with parameter" },
  ];
}

export default function DynamicPage({ params }: Route.ComponentProps) {
  const { name } = params;
  const [count, setCount] = useState(0);
  
  return (
    <main>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
      <div>
        <p>Count: {count}</p>
        <button onClick={() => setCount(count + 1)}>Increment</button>
      </div>
    </main>
  );
}
