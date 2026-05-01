import React from 'react';

const HelloWorldPage = async ({
  params,
}: {
  params: Promise<{ name: string }>;
}) => {
  const { name } = await params;
  return (
    <div>
      <h1>Hello World</h1>
      <p>Dynamic ID: {name}</p>
    </div>
  );
};

export default HelloWorldPage;