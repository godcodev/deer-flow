// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

"use client";

import Mention from "@tiptap/extension-mention";
import { Editor, Extension, type Content } from "@tiptap/react";
import {
  EditorContent,
  type EditorInstance,
  EditorRoot,
  type JSONContent,
  StarterKit,
  Placeholder,
} from "novel";
import { Markdown } from "tiptap-markdown";
import { useDebouncedCallback } from "use-debounce";

import "~/styles/prosemirror.css";
import { resourceSuggestion } from "./resource-suggestion";
import React, { forwardRef, useMemo, useRef } from "react";
import type { Resource } from "~/core/messages";
import { useRAGProvider } from "~/core/api/hooks";
import { LoadingOutlined } from "@ant-design/icons";

export interface MessageInputRef {
  focus: () => void;
  submit: () => void;
}

export interface MessageInputProps {
  className?: string;
  placeholder?: string;
  onChange?: (markdown: string) => void;
  onEnter?: (message: string, resources: Array<Resource>) => void;
}

function formatMessage(content: JSONContent) {
  if (content.content) {
    const output: {
      text: string;
      resources: Array<Resource>;
    } = {
      text: "",
      resources: [],
    };
    for (const node of content.content) {
      const { text, resources } = formatMessage(node);
      output.text += text;
      output.resources.push(...resources);
    }
    return output;
  } else {
    return formatItem(content);
  }
}

function formatItem(item: JSONContent): {
  text: string;
  resources: Array<Resource>;
} {
  if (item.type === "text") {
    return { text: item.text ?? "", resources: [] };
  }
  if (item.type === "mention") {
    return {
      text: `[${item.attrs?.label}](${item.attrs?.id})`,
      resources: [
        { uri: item.attrs?.id ?? "", title: item.attrs?.label ?? "" },
      ],
    };
  }
  return { text: "", resources: [] };
}

const MessageInput = forwardRef<MessageInputRef, MessageInputProps>(
  ({ className, onChange, onEnter }: MessageInputProps, ref) => {
    const editorRef = useRef<Editor>(null);
    const debouncedUpdates = useDebouncedCallback(
      async (editor: EditorInstance) => {
        if (onChange) {
          const markdown = editor.storage.markdown.getMarkdown();
          onChange(markdown);
        }
      },
      200,
    );

    React.useImperativeHandle(ref, () => ({
      focus: () => {
        editorRef.current?.view.focus();
      },
      submit: () => {
        if (onEnter) {
          const { text, resources } = formatMessage(
            editorRef.current?.getJSON() ?? [],
          );
          onEnter(text, resources);
        }
      },
    }));

    const { provider, loading } = useRAGProvider();

    const extensions = useMemo(() => {
      const extensions = [
        StarterKit,
        Markdown.configure({
          html: true,
          tightLists: true,
          tightListClass: "tight",
          bulletListMarker: "-",
          linkify: false,
          breaks: false,
          transformPastedText: false,
          transformCopiedText: false,
        }),
        Placeholder.configure({
          showOnlyCurrent: false,
          placeholder: provider
            ? "What can I do for you? \nYou may refer to RAG resources by using @."
            : "What can I do for you?",
          emptyEditorClass: "placeholder",
        }),
        Extension.create({
          name: "keyboardHandler",
          addKeyboardShortcuts() {
            return {
              Enter: () => {
                if (onEnter) {
                  const { text, resources } = formatMessage(
                    this.editor.getJSON() ?? [],
                  );
                  onEnter(text, resources);
                }
                return this.editor.commands.clearContent();
              },
            };
          },
        }),
      ];
      if (provider) {
        extensions.push(
          Mention.configure({
            HTMLAttributes: {
              class: "mention",
            },
            suggestion: resourceSuggestion,
          }) as Extension,
        );
      }
      return extensions;
    }, [onEnter, provider]);

    if (loading) {
      return (
        <div className={className}>
          <LoadingOutlined />
        </div>
      );
    }

    return (
      <div className={className}>
        <EditorRoot>
          <EditorContent
            immediatelyRender={false}
            extensions={extensions}
            className="border-muted h-full w-full overflow-auto"
            editorProps={{
              attributes: {
                class:
                  "prose prose-base dark:prose-invert inline-editor font-default focus:outline-none max-w-full",
              },
            }}
            onCreate={({ editor }) => {
              editorRef.current = editor;
            }}
            onUpdate={({ editor }) => {
              debouncedUpdates(editor);
            }}
          ></EditorContent>
        </EditorRoot>
      </div>
    );
  },
);

export default MessageInput;
