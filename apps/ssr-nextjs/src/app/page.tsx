"use client";

import { useState } from 'react';

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
