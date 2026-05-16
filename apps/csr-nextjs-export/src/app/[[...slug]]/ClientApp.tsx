"use client";

import React, { useState, useEffect } from 'react';
import { usePathname } from 'next/navigation';

export default function ClientApp() {
  const [count, setCount] = useState(0);
  const pathname = usePathname();
  const [slug, setSlug] = useState<string[]>([]);

  useEffect(() => {
    if (pathname) {
      const parts = pathname.split('/').filter(Boolean);
      setSlug(parts);
    }
  }, [pathname]);

  // Home Route: / (slug is empty)
  if (slug.length === 0) {
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

  // Dynamic Route: /dynamic/[name]
  if (slug[0] === 'dynamic' && slug[1]) {
    const name = slug[1];
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

  // Fallback for SPA (404)
  return (
    <main>
      <h1>404 - Not Found</h1>
      <p>Path: {pathname}</p>
    </main>
  );
}
