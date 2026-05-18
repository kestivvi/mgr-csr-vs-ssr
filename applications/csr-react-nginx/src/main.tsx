import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { createBrowserRouter, RouterProvider } from 'react-router-dom';
import App from './App.tsx'
import Dynamic from './pages/Dynamic.tsx';
import DynamicApp from './pages/DynamicApp.tsx';
import './global.css';

const router = createBrowserRouter([
  {
    path: "/",
    element: <App />,
  },
  {
    path: "/dynamic/:name",
    element: <Dynamic />,
  },
  {
    path: "/dynamic-app/:id",
    element: <DynamicApp />,
  },
]);

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
)
