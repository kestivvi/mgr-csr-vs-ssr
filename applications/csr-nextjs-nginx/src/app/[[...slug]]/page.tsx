import React from 'react';
import ClientApp from './ClientApp';

// For static export, we generate only the root index.html.
// Nginx handles the rest by falling back to index.html.
export function generateStaticParams() {
  return [{ slug: [] }];
}

export default function Page() {
  return <ClientApp />;
}
