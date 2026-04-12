import { createFileRoute } from "@tanstack/react-router";
import { Thread } from "@assistant-ui/react-ui";
import { MyRuntimeProvider } from "@/components/MyRuntimeProvider";

export const Route = createFileRoute("/_layout/chatbot")({
  component: ChatbotPage,
});

function ChatbotPage() {
  return (
    <MyRuntimeProvider>
      <div className="flex h-[calc(100vh-theme(spacing.16))] flex-col">
        <Thread />
      </div>
    </MyRuntimeProvider>
  );
}
