import { type ReactNode } from "react";
import { OpenAPI } from "@/client";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { useChatRuntime, AssistantChatTransport } from "@assistant-ui/react-ai-sdk";
import { ThreadList } from "@/components/assistant-ui/thread-list";
import { Thread } from "@/components/assistant-ui/thread";

export function MyRuntimeProvider({children}: { children: ReactNode }) {

    const runtime = useChatRuntime({
        transport: new AssistantChatTransport({
            api: `${OpenAPI.BASE}/api/v1/chat/stream`,
          headers: async () => {
            const token = await (typeof OpenAPI.TOKEN === 'function' ? OpenAPI.TOKEN() : OpenAPI.TOKEN);
            return {
                Authorization: `Bearer ${token}`,
            };
        }
        }),
    });
    return (
        <AssistantRuntimeProvider runtime={runtime}>
            <div className="flex h-full w-full">
                <ThreadList/>
                <div className="flex-grow">
                  <Thread/>
                </div>
                {children}
            </div>
        </AssistantRuntimeProvider>
    );
}
