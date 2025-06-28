import React from 'react';

const HelloWorldPage = async ({
  params,
}: {
  params: Promise<{ name: string }>;
}) => {
  const { name } = await params;
  return (
    <div>
      <h1>Hello, {name}</h1>
    </div>
  );
};

export default HelloWorldPage;