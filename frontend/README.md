# xagent Frontend

A modern React-based frontend for the xagent Agent system, built with Next.js, TypeScript, and Tailwind CSS.

## Features

- **Three-Column Layout**: User interaction, DAG visualization, and step details
- **Real-time Communication**: WebSocket integration for live updates
- **DAG Visualization**: Interactive graph execution display using React Flow
- **Modern UI**: Built with shadcn/ui components and Tailwind CSS
- **TypeScript**: Full type safety throughout the application
- **Responsive Design**: Works on desktop and mobile devices

## Tech Stack

- **Framework**: Next.js 15 with App Router
- **Language**: TypeScript
- **Styling**: Tailwind CSS
- **UI Components**: shadcn/ui
- **State Management**: React Context + useReducer
- **Real-time**: Socket.io Client
- **Visualization**: @xyflow/react (React Flow)
- **Markdown**: Marked.js + Highlight.js
- **Icons**: Lucide React

## Getting Started

### Prerequisites

- Node.js 20.19+ within the 20.x line, or 22.12+ (required by the Vite 7 test toolchain)
- npm or yarn

### Installation

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Configure environment variables:
```bash
cp .env.local.example .env.local
```

4. Start the development server:
```bash
npm run dev
```

5. Open [http://localhost:3000](http://localhost:3000) in your browser.

## Project Structure

```
src/
├── app/                    # Next.js app router
│   ├── globals.css        # Global styles
│   ├── layout.tsx         # Root layout
│   └── page.tsx           # Main application page
├── components/            # React components
│   ├── layout/           # Layout components
│   │   ├── three-column-layout.tsx
│   │   ├── left-panel.tsx
│   │   ├── center-panel.tsx
│   │   └── right-panel.tsx
│   └── ui/               # shadcn/ui components
├── contexts/             # React contexts
│   └── app-context.tsx   # Global app state
├── hooks/               # Custom React hooks
│   └── use-websocket.ts  # WebSocket hook
└── lib/                 # Utility functions
    ├── utils.ts         # General utilities
    └── markdown.ts      # Markdown rendering
```

## Key Components

### ThreeColumnLayout
Main layout component that arranges the three panels.

### LeftPanel
Handles user input and displays conversation history.

### CenterPanel
Interactive DAG visualization using React Flow.

### RightPanel
Shows step execution details and trace events.

### WebSocket Integration
Real-time communication with the FastAPI backend for live updates.

## Configuration

### Environment Variables

- `NEXT_PUBLIC_WS_URL`: WebSocket server URL
- `NEXT_PUBLIC_API_URL`: HTTP API server URL
- `NEXT_PUBLIC_XAGENT_HTML_PREVIEW_ORIGIN`: Optional isolated HTML preview origin.
  Set an HTTPS origin (HTTP is allowed only on localhost/127.0.0.1/[::1] for development)
  to load its `/render` runner; any configured path/query/fragment is discarded. Use `auto` for the XAgent SaaS
  mapping (`cloud.xagent.co` to `preview.xagent.co`, stage/dev to
  `preview-stage.xagent.co` / `preview-dev.xagent.co`). Unset/invalid settings or an 8-second ready timeout use
  the scripts-only opaque `srcDoc` fallback. The serving CSP must permit the
  runner in `frame-src`: Next includes the build setting; packaged/static
  serving needs the same variable at backend runtime (and custom proxies must
  match). Docker accepts a build argument. Nightly builds use the repository
  variable of the same name, defaulting to disabled; the official PyPI wheel is
  built with the runner disabled and needs a custom frontend build to enable it.
  Setting a runtime variable on an already-built frontend does not enable it.
  The separate runner is not shipped or deployed by this repository. It must
  implement the `/render` ready/render v1 protocol described in
  [HTML preview security](../docs/html-preview-security.md).
  Embedded widget/share pages retain `srcDoc`, because the managed runner's
  exact `frame-ancestors` policy cannot authorize an arbitrary outer customer
  site. Static quoted `src`/`href` assets are rewritten; other relative URL
  syntaxes and runtime-constructed URLs must be absolute or inlined.

### Backend Integration

The frontend connects to the FastAPI backend running on port 8000 by default. Make sure the backend is running before starting the frontend.

## Development

### Available Scripts

- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run start` - Start production server
- `npm run lint` - Run ESLint
- `npm run type-check` - Run TypeScript type checking

### Adding New Components

1. Use shadcn/ui for consistent design:
```bash
npx shadcn@latest add [component-name]
```

2. Follow the established patterns in the existing components.

### Styling

- Use Tailwind CSS classes for styling
- Follow the shadcn/ui design system
- Use the `cn()` utility for conditional class merging

## WebSocket Events

The frontend handles the following WebSocket events:

- `trace_event` - Execution trace events
- `task_completed` - Task completion notifications
- `dag_execution` - DAG execution state updates
- `dag_step_info` - Individual step information
- `task_paused` / `task_resumed` - Task state changes
- `agent_error` - Error notifications

## Browser Support

- Chrome 90+
- Firefox 96+
- Safari 15.4+
- Edge 90+

Browser sign-in also requires local storage, the Web Locks API, and a secure
context. Use HTTPS outside local development; loopback hosts such as
`localhost`, `127.0.0.1`, and `::1` are suitable for local development.

## Contributing

1. Follow the existing code style
2. Add TypeScript types for new features
3. Test thoroughly
4. Update documentation as needed

## License

This project is part of xagent and follows the same license terms.
