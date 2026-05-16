"use client";

import React, { useState, use } from 'react';

const HelloWorldPage = ({
  params,
}: {
  params: Promise<{ name: string }>;
}) => {
  const { name } = use(params);
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
};

export default HelloWorldPage;