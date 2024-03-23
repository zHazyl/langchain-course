from flask import Blueprint, g, request, Response, jsonify, stream_with_context
from app.web.hooks import login_required, load_model
from app.web.db.models import Pdf, Conversation
from app.chat import build_chat, ChatArgs, build_summary
from app.chat.llms.chatopenai import build_classify_llm

bp = Blueprint("conversation", __name__, url_prefix="/api/conversations")


@bp.route("/", methods=["GET"])
@login_required
@load_model(Pdf, lambda r: r.args.get("pdf_id"))
def list_conversations(pdf):
    return [c.as_dict() for c in pdf.conversations]


@bp.route("/", methods=["POST"])
@login_required
@load_model(Pdf, lambda r: r.args.get("pdf_id"))
def create_conversation(pdf):
    conversation = Conversation.create(user_id=g.user.id, pdf_id=pdf.id)

    return conversation.as_dict()


@bp.route("/<string:conversation_id>/messages", methods=["POST"])
@login_required
@load_model(Conversation)
def create_message(conversation):
    input = request.json.get("input")
    streaming = request.args.get("stream", False)

    pdf = conversation.pdf

    chat_args = ChatArgs(
        conversation_id=conversation.id,
        pdf_id=pdf.id,
        streaming=streaming,
        metadata={
            "conversation_id": conversation.id,
            "user_id": g.user.id,
            "pdf_id": pdf.id,
        },
    )

    classify_llm = build_classify_llm()
    type_question = classify_llm.invoke(f"""
        Classify the following question into categories:
        - Summary return: summary
        - Question or request return: question/request
        - Generate quiz: quiz
        - Else: question/request

        For example:
        Summarize this doc
        summary

        please help me summarize history
        summary

        What is Spring used for
        question/request

        Show me more about it
        question/request

        Generate quiz
        quiz
                                        
        As table format
        question/request

        ===============

        {input}
    """).content

    chat = None

    print(type_question)

    if type_question == "question/request":
        chat = build_chat(chat_args)

    elif type_question == "summary":
        print("type_summary")
        print(input)
        chat = build_summary(chat_args)

    elif type_question == "quiz":
        print("quiz")

    if not chat:
        return type_question.__str__()

    if streaming:
        return Response(
            stream_with_context(chat.stream(input)), mimetype="text/event-stream"
        )
    else:
        return jsonify({"role": "assistant", "content": chat.run(input)})
