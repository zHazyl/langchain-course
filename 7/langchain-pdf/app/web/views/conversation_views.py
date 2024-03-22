from flask import Blueprint, g, request, Response, jsonify, stream_with_context
from app.web.hooks import login_required, load_model
from app.web.db.models import Pdf, Conversation
from app.chat import build_chat, ChatArgs
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
        Classify the following question into 4 category:
        - Overall summary return: overall
        - Summary from page to page return: range,start,end
        - Question return: question/request
        - Generate quiz range: quiz,start,end

        For example:
        Summary this doc
        overall

        Summary page 1 to 10
        range,1,10

        What is Spring used for
        question/request

        Show me more about it
        question/request

        Generate quiz page 1 to 10
        quiz,1,10

        ===============

        {input}
    """).content.split(',')

    chat = None
    print(chat_args)


    if type_question[0] == "question/request":
        chat = build_chat(chat_args)

    elif type_question[0] == "overall":
        print("overall")
        print(chat_args)

    elif type_question[0] == "range":
        print("range")

    elif type_question[0] == "quiz":
        print("quiz")

    if not chat:
        return "Chat not yet implemented!"

    if streaming:
        return Response(
            stream_with_context(chat.stream(input)), mimetype="text/event-stream"
        )
    else:
        return jsonify({"role": "assistant", "content": chat.run(input)})
